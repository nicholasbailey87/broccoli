import warnings
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from einops import rearrange

from .rope import RotaryEmbedding, apply_rotary_emb

try:
    from flash_attn import flash_attn_func

    print("Using flash-attn.")
    FLASH_ATTN = True
except ImportError:
    pass
    FLASH_ATTN = False


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """
    From https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    Copyright 2019 Ross Wightman
    See documentation and licence there.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    From https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    Copyright 2019 Ross Wightman
    See documentation and licence there.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class MHAttention(nn.Module):
    """
    Multi-head self-attention using einops and optionally a custom linear layer.

    Forward method assumes q, k and v have the same embedding size and k and v
        are the same shape.

    Assumes bias=False and batch_first=True, as God intended.
    """

    def __init__(
        self,
        embed_dim,
        n_heads,
        dropout=0.0,
        causal=False,
        seq_len=None,
        linear_module: nn.Module = nn.Linear,
        utility_tokens=0,
        talking_heads=False,
        rotary_embedding=None,
        source_size=None,
        scaling="d",
    ):
        """
        Args:
            scaling: how should the attention logits be scaled? Can be "sqrtd"
                to mimic the original Attention is All You Need approach of
                dividing by the sqrt of the embedding Dimension or "d" per
                "Tensor Programs V...". Default "d"
        """
        super().__init__()

        if rotary_embedding is not None:
            assert source_size is not None
        if causal:
            assert seq_len is not None

        self.talking_heads = talking_heads

        if self.talking_heads:
            self.head_projection = nn.Linear(n_heads, n_heads, bias=False)
            self.sample_projection = nn.Linear(n_heads, n_heads, bias=False)
        else:
            self.head_projection = None
            self.sample_projection = None

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0
        self.scaling = scaling

        self.head_dim = self.embed_dim // self.n_heads

        if self.scaling == "sqrtd":
            self.scaling_factor = 1 / math.sqrt(self.head_dim)
        elif self.scaling == "d":
            # 8/d_model for backwards compatibility,
            #     per https://github.com/microsoft/mup
            self.scaling_factor = 8 / self.head_dim
        else:
            raise ValueError('`scaling` argument to MHAttention must be "d" or "sqrtd"')

        self.q_proj = linear_module(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = linear_module(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = linear_module(self.embed_dim, self.embed_dim, bias=False)

        self.out_proj = linear_module(self.embed_dim, self.embed_dim, bias=False)

        self.causal = causal
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        if self.causal:
            self.register_buffer(
                "mask",
                (torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 1)
                .unsqueeze(0)
                .unsqueeze(0),
            )
        self.rotary_embedding = rotary_embedding
        self.source_size = source_size
        self.utility_tokens = utility_tokens

        self.reset_parameters()

    @property
    def _kv_distance(self) -> float:
        """
        Calculates the cosine distance between the weight tensors of `self.k_proj`
            and `self.v_proj`.

        The cosine distance is defined as 1 - cosine_similarity (i.e. a value
            closer to 0 indicates higher similarity.
        """

        similarity = F.cosine_similarity(
            self.k_proj.weight.detach().flatten(),
            self.v_proj.weight.detach().flatten(),
            dim=0,
            eps=1e-8,
        ).item()

        return 1 - similarity

    def add_axial_rope(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Axial RoPE to all tokens except utility tokens
        """

        if len(self.source_size) == 1:
            spatial_dimension_names = "D1"
            spatial_dimension_values = {"D1": self.source_size[0]}
        elif len(self.source_size) == 2:
            spatial_dimension_names = "D1 D2"
            spatial_dimension_values = {
                "D1": self.source_size[0],
                "D2": self.source_size[1],
            }
        elif len(self.source_size) == 3:
            spatial_dimension_names = "D1 D2 D3"
            spatial_dimension_values = {
                "D1": self.source_size[0],
                "D2": self.source_size[1],
                "D3": self.source_size[2],
            }
        else:
            raise NotImplementedError(
                "`source_size` must be a tuple of 1, 2 or 3 integers"
            )

        q_util, q_img = q[:, : self.utility_tokens, :], q[:, self.utility_tokens :, :]
        k_util, k_img = k[:, : self.utility_tokens, :], k[:, self.utility_tokens :, :]

        q_img = rearrange(
            q_img,
            f"b ({spatial_dimension_names}) d -> b {spatial_dimension_names} d",
            **spatial_dimension_values,
        )
        k_img = rearrange(
            k_img,
            f"b ({spatial_dimension_names}) d -> b {spatial_dimension_names} d",
            **spatial_dimension_values,
        )

        freqs = self.rotary_embedding.get_axial_freqs(*self.source_size)

        q_img = apply_rotary_emb(freqs, q_img)
        k_img = apply_rotary_emb(freqs, k_img)

        q_img = rearrange(
            q_img,
            f"b {spatial_dimension_names} d -> b ({spatial_dimension_names}) d",
        )
        k_img = rearrange(
            k_img,
            f"b {spatial_dimension_names} d -> b ({spatial_dimension_names}) d",
        )

        # Re-combine the utility tokens and the RoPE-enhanced sequence tokens
        q = torch.cat([q_util, q_img], dim=1)
        k = torch.cat([k_util, k_img], dim=1)

        return q, k

    def project_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_batch_size, query_tokens, query_features = q.size()
        key_batch_size, key_tokens, key_features = k.size()

        assert k.size() == v.size()
        assert query_features == key_features
        assert (
            (query_batch_size == key_batch_size)  # batch sizes are the same...
            or query_batch_size == 1  # ... or query is broadcastable
        )

        if self.causal:
            assert query_tokens == key_tokens
            assert query_tokens == self.seq_len

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        if self.rotary_embedding is not None:
            q, k = self.add_axial_rope(q, k)

        return q, k, v

    def forward(self, q, k, v):

        q, k, v = self.project_qkv(q, k, v)

        if FLASH_ATTN and not self.talking_heads:
            # Divide Q/K/V into heads
            q = rearrange(q, "b t (h d) -> b t h d", h=self.n_heads)
            k = rearrange(k, "b t (h d) -> b t h d", h=self.n_heads)
            v = rearrange(v, "b t (h d) -> b t h d", h=self.n_heads)

            output_with_heads = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=self.scaling_factor,
                causal=self.causal,
            )

            output_without_heads = rearrange(output_with_heads, "b t h d -> b t (h d)")

            return self.out_proj(output_without_heads)
        else:
            # Divide Q/K/V into heads
            q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)
            k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
            v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)

            qk_scores = q @ k.transpose(-1, -2)

            qk_scores *= self.scaling_factor

            if self.talking_heads:
                qk_scores = torch.einsum(
                    "b h i j, o h -> b o i j", qk_scores, self.head_projection.weight
                )

            # Apply mask if causal (must come before softmax)
            if self.causal:
                qk_scores.masked_fill_(self.mask, float("-inf"))

            qk_scores = F.softmax(qk_scores, dim=-1)

            if self.talking_heads:
                qk_scores = torch.einsum(
                    "b h i j, o h -> b o i j", qk_scores, self.sample_projection.weight
                )

            qk_scores = self.dropout(qk_scores)

            output_with_heads = qk_scores @ v

            output_without_heads = rearrange(output_with_heads, "b h t d -> b t (h d)")

            return self.out_proj(output_without_heads)

    def attention_logits(self, q, k, v):

        q, k, v = self.project_qkv(q, k, v)

        # Divide Q/K/V into heads
        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)

        qk_scores = q @ k.transpose(-1, -2)

        qk_scores *= self.scaling_factor

        # Apply mask if causal (must come before softmax)
        if self.causal:
            qk_scores.masked_fill_(self.mask, float("-inf"))

        return qk_scores  # (batch, head, seq_len, seq_len)

    def reset_parameters(self):
        # Default nn.Linear init is kaiming_uniform, which is fine
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.out_proj.reset_parameters()
        if self.talking_heads:
            # Initialize close to identity
            nn.init.eye_(self.head_projection.weight)
            nn.init.eye_(self.sample_projection.weight)


class FeedforwardBlock(nn.Module):
    """
    ...
    """

    def __init__(
        self,
        input_features,
        ratio,
        output_features,
        activation=nn.ReLU,
        activation_kwargs=None,
        dropout=0.0,
        inner_dropout=None,
        outer_dropout=None,
        linear_module_up=nn.Linear,
        linear_module_down=nn.Linear,
        pre_norm=True,
        normformer=False,
        post_norm=True,
        residual_path=True,
        checkpoint=True,
    ):
        super().__init__()

        self.checkpoint = checkpoint
        self.residual_path = residual_path
        self.post_norm = post_norm
        self.xglu = activation.__name__.endswith("GLU")

        if self.residual_path and (output_features < input_features):
            raise ValueError(
                "If the number of output features will be less than "
                "the number of input features, then `residual_path` "
                "should be set to False."
            )

        if self.post_norm:
            self.layernorm = nn.LayerNorm(output_features)

        if activation_kwargs is not None:
            self.activation = activation(**activation_kwargs)
        else:
            self.activation = activation()

        self.inner_dropout = nn.Dropout(
            inner_dropout if inner_dropout is not None else dropout
        )
        self.outer_dropout = nn.Dropout(
            outer_dropout if outer_dropout is not None else dropout
        )

        self.max_features = (
            2 * ratio * output_features if self.xglu else ratio * output_features
        )

        self.linear_in = linear_module_up(input_features, self.max_features)
        self.linear_out = linear_module_down(ratio * output_features, output_features)

        self.process = nn.Sequential(
            *[
                nn.LayerNorm(input_features) if pre_norm else nn.Identity(),
                self.linear_in,
                self.activation,
                self.inner_dropout,
                nn.LayerNorm(ratio * output_features) if normformer else nn.Identity(),
                self.linear_out,
                self.outer_dropout,
            ]
        )

        self.recycling_enabled = False
        if hasattr(self.linear_in, "row_recycling_rate") and hasattr(
            self.linear_out, "column_recycling_rate"
        ):
            self.recycling_enabled = True
            self.master_recycling_rate = self.linear_in.row_recycling_rate
            self.linear_in.row_recycling_rate = 0.0
            self.linear_out.column_recycling_rate = 0.0
            if (
                hasattr(self.linear_in, "column_recycling_rate")
                and self.linear_in.column_recycling_rate > 0
            ) or (
                hasattr(self.linear_out, "row_recycling_rate")
                and self.linear_out.row_recycling_rate > 0
            ):
                raise NotImplementedError(
                    "At the moment this layer can only support recycling linear "
                    "layers if the in layer resets only rows and the out layer "
                    "resets only columns."
                )

        self.reset_parameters()

    def forward(self, x):

        # Recycle weights if using recycling linear layers
        if self.training and self.recycling_enabled:
            indices = self.linear_out.get_reset_indices(1)
            self.linear_in.reset_rows(indices, incoming_data=x)
            self.linear_out.reset_columns(indices)

        if self.checkpoint:
            processed = checkpoint(self.process, x, use_reentrant=False)
        else:
            processed = self.process(x)

        if self.residual_path and self.post_norm:
            return self.layernorm(x + processed)
        elif self.residual_path:
            return x + processed
        else:
            return processed

    def reset_parameters(self):
        if self.post_norm:
            self.layernorm.reset_parameters()

        # Iterate over the sequential block to reset parameters
        for module in self.process:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


class TransformerBlock(nn.Module):
    """
    Performs LayerNorms first (as in PyTorch Transformers when norm_first=True),
        which is also what is seen in e.g.
        https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        and is recommended by https://arxiv.org/abs/2002.04745

    """

    def __init__(
        self,
        seq_len,
        d_model,
        n_heads,
        relative_position_embedding=False,
        source_size=None,
        utility_tokens=0,
        talking_heads=False,
        mlp_ratio=4,
        activation: nn.Module = nn.ReLU,
        activation_kwargs: Optional[dict] = None,
        ff_linear_module_up=None,
        ff_linear_module_down=None,
        msa_scaling="d",
        ff_dropout=0.0,
        ff_inner_dropout=0.0,
        ff_outer_dropout=0.0,
        msa_dropout=0.0,
        identity_probability=0.0,
        causal=False,
        linear_module=nn.Linear,
        pre_norm=True,
        post_norm=False,
        normformer=False,
        checkpoint_ff=True,
    ):
        """
        Args:
            msa_scaling: how should the attention logits be scaled? Can be "sqrtd"
                to mimic the original Attention is All You Need approach of
                dividing by the sqrt of the embedding Dimension or "d" per
                "Tensor Programs V...". Default "d"
        """

        super().__init__()

        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.normformer = normformer

        self.drop_path = DropPath(drop_prob=identity_probability, scale_by_keep=True)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)

        if relative_position_embedding:
            max_freq = int(max(source_size) / 2)  # Suggested by Gemini!
            if d_model < 16:
                dim = d_model
            else:
                dim = 16
            self.rotary_embedding = RotaryEmbedding(
                dim=dim, freqs_for="pixel", max_freq=max_freq
            )
        else:
            self.rotary_embedding = None

        self.attn = MHAttention(  # Handles QKV projection
            d_model,
            n_heads,
            dropout=msa_dropout,
            causal=causal,
            seq_len=seq_len,
            linear_module=linear_module,
            rotary_embedding=self.rotary_embedding,
            source_size=source_size,
            utility_tokens=utility_tokens,
            talking_heads=talking_heads,
            scaling=msa_scaling,
        )

        # Submodule for the feedforward process
        self.ff = FeedforwardBlock(
            d_model,
            mlp_ratio,
            d_model,
            activation=activation,
            activation_kwargs=activation_kwargs,
            dropout=ff_dropout,
            inner_dropout=ff_inner_dropout,
            outer_dropout=ff_outer_dropout,
            linear_module_up=(
                ff_linear_module_up
                if ff_linear_module_up is not None
                else linear_module
            ),
            linear_module_down=(
                ff_linear_module_down
                if ff_linear_module_down is not None
                else linear_module
            ),
            pre_norm=False,  # Handled outside the block
            normformer=normformer,
            post_norm=False,  # Handled outside the block
            residual_path=False,  # Handled outside the block
            checkpoint=checkpoint_ff,
        )

        self.reset_parameters()

    @property
    def _kv_distance(self) -> float:
        return self.attn._kv_distance

    def forward(self, x):

        if self.pre_norm:
            x = self.layer_norm_1(x)
            x = x + self.drop_path(self.attn(x, x, x))
            x = self.layer_norm_2(x)
            x = x + self.drop_path(self.ff(x))
            if self.post_norm:  # i.e. in addition! Pre and post.
                x = self.layer_norm_3(x)
        elif self.post_norm:  # i.e. only, not prenorm, just post
            x = x + self.drop_path(self.attn(x, x, x))
            x = self.layer_norm_1(x)
            x = x + self.drop_path(self.ff(x))
            x = self.layer_norm_2(x)
        else:  # Not pre or post norm. Stand well back.
            x = x + self.drop_path(self.attn(x, x, x))
            x = x + self.drop_path(self.ff(x))

        return x

    def attention_logits(self, x):
        """
        Give back the attention scores used in this layer.
        """
        if self.pre_norm:
            x = self.layer_norm_1(x)
            return self.attn.attention_logits(x, x, x)
        else:
            return self.attn.attention_logits(x, x, x)

    def reset_parameters(self):
        self.layer_norm_1.reset_parameters()
        self.layer_norm_2.reset_parameters()
        self.layer_norm_3.reset_parameters()

        self.attn.reset_parameters()
        self.ff.reset_parameters()


class TransformerEncoder(nn.Module):
    """
    This assumes we already get a sequence of embeddings (e.g. word or image
        patch embeddings).
    """

    def __init__(
        self,
        seq_len,
        d_model,
        n_layers,
        n_heads,
        absolute_position_embedding=True,
        relative_position_embedding=False,
        source_size=None,
        mlp_ratio=4,
        activation: nn.Module = nn.ReLU,
        activation_kwargs: Optional[dict] = None,
        ff_linear_module_up=None,
        ff_linear_module_down=None,
        ff_dropout=0.0,
        ff_inner_dropout=0.0,
        ff_outer_dropout=0.0,
        msa_dropout=0.0,
        stochastic_depth=0.0,
        causal=False,
        linear_module=nn.Linear,
        utility_tokens=0,
        talking_heads=False,
        return_utility_tokens=False,
        pre_norm=True,
        post_norm=False,
        normformer=False,
        msa_scaling="d",
        checkpoint_ff=True,
    ):
        """
        Args:
            msa_scaling: how should the attention logits be scaled? Can be "sqrtd"
                to mimic the original Attention is All You Need approach of
                dividing by the sqrt of the embedding Dimension or "d" per
                "Tensor Programs V...". Default "d"
        """

        if relative_position_embedding and (source_size is None):
            raise ValueError(
                "`source_size` for TransformerEncoder cannot be None if"
                " `relative_position_embedding` is True"
            )

        if absolute_position_embedding and (seq_len is None):
            raise ValueError(
                "`seq_len` for TransformerEncoder cannot be None if"
                " `absolute_position_embedding` is True"
            )

        super().__init__()

        if FLASH_ATTN and talking_heads:
            warnings.warn(
                "Using talking heads currently prevents using flash attention.",
                stacklevel=2,
            )

        self.seq_len = seq_len
        self.n_heads = n_heads
        self._utility_tokens = utility_tokens
        self.return_utility_tokens = return_utility_tokens

        # Initialise utility tokens with normal init, like usual Pytorch embeddings
        if self._utility_tokens:
            self._utility_token_embedding = nn.Parameter(
                torch.empty(self._utility_tokens, d_model)
            )
            nn.init.normal_(self._utility_token_embedding, mean=0.0, std=1.0)
        else:
            self._utility_token_embedding = None

        if self._utility_tokens and (self.seq_len is not None):
            self.full_sequence_length = self.seq_len + self._utility_tokens
        else:
            self.full_sequence_length = self.seq_len

        self.d_model = d_model

        if absolute_position_embedding:
            self.absolute_position_embedding = nn.Embedding(
                self.full_sequence_length, d_model
            )
        else:
            self.absolute_position_embedding = None

        self.mlp_dropout = ff_dropout
        self.msa_dropout = msa_dropout
        self.stochastic_depth = stochastic_depth

        assert isinstance(n_layers, int)

        if n_layers == 1:
            self.stochastic_depth_probabilities = [0.0]
        else:
            step_size = self.stochastic_depth / (n_layers - 1)
            self.stochastic_depth_probabilities = [
                i * step_size for i in range(n_layers)
            ]

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.full_sequence_length,
                    d_model,
                    n_heads,
                    relative_position_embedding=relative_position_embedding,
                    source_size=source_size,
                    utility_tokens=utility_tokens,
                    talking_heads=talking_heads,
                    mlp_ratio=mlp_ratio,
                    activation=activation,
                    activation_kwargs=activation_kwargs,
                    ff_linear_module_up=ff_linear_module_up,
                    ff_linear_module_down=ff_linear_module_down,
                    msa_scaling=msa_scaling,
                    ff_dropout=ff_dropout,
                    ff_inner_dropout=ff_inner_dropout,
                    ff_outer_dropout=ff_outer_dropout,
                    msa_dropout=msa_dropout,
                    identity_probability=self.stochastic_depth_probabilities[i],
                    causal=causal,
                    linear_module=linear_module,
                    pre_norm=pre_norm,
                    post_norm=post_norm,
                    normformer=normformer,
                    checkpoint_ff=checkpoint_ff,
                )
                for i in range(n_layers)
            ]
        )

        self.reset_parameters()

    @property
    def _kv_distances(self) -> float:
        return ",".join([str(block._kv_distance) for block in self.blocks])

    def preprocess(self, x):
        if self._utility_tokens:
            x = torch.cat(
                [self._utility_token_embedding.expand(x.size(0), -1, -1), x], dim=1
            )
        else:
            x = x

        if self.absolute_position_embedding is not None:
            x = x + self.absolute_position_embedding(
                torch.arange(
                    0, self.full_sequence_length, dtype=torch.long, device=x.device
                ).unsqueeze(
                    0
                )  # to shape (1, seq_len) to broadcast over batch
            )

        return x

    def forward(self, x):

        x = self.preprocess(x)

        for block in self.blocks:
            x = block(x)

        if self._utility_tokens and not self.return_utility_tokens:
            return x[:, self._utility_tokens :, :]
        else:
            return x

    def attention_logits(self, x):

        x = self.preprocess(x)

        layer_scores = []

        for block in self.blocks:
            # Get attention scores with shape (batch, 1, head, seq_len, seq_len)
            layer_attention_logits = block.attention_logits(x).unsqueeze(1)
            layer_scores.append(layer_attention_logits)
            x = block(x)

        return torch.cat(layer_scores, dim=1)  # (batch, layer, head, seq_len, seq_len)

    def reset_parameters(self):
        if self._utility_token_embedding is not None:
            nn.init.normal_(self._utility_token_embedding, mean=0.0, std=1.0)

        if self.absolute_position_embedding is not None:
            self.absolute_position_embedding.reset_parameters()

        for block in self.blocks:
            block.reset_parameters()
