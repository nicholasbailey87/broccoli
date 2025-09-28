import math
from collections import OrderedDict
from typing import Optional
from numpy import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .rope import RotaryEmbedding, apply_rotary_emb
from .linear import AnchoredLinear, SpectralNormLinear


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
        bos_tokens=0,
        rotary_embedding=None,
        source_size=None,
    ):
        super().__init__()

        if rotary_embedding is not None:
            assert source_size is not None
        if causal:
            assert seq_len is not None

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0

        self.head_dim = self.embed_dim // self.n_heads

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
        self.bos_tokens = bos_tokens

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

    def forward(self, q, k, v):
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
            assert query_tokens == self.sequence_length

        # Project q, k and v
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Rearrange dimensions and add RoPE if needed
        if self.rotary_embedding is not None:

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

            q_bos, q_img = q[:, : self.bos_tokens, :], q[:, self.bos_tokens :, :]
            k_bos, k_img = k[:, : self.bos_tokens, :], k[:, self.bos_tokens :, :]

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

            # Re-combine the BOS tokens and the RoPE-enhanced image tokens
            q = torch.cat([q_bos, q_img], dim=1)
            k = torch.cat([k_bos, k_img], dim=1)

        # Divide Q/K/V into heads
        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)

        qk_scores = q @ k.transpose(-1, -2)

        qk_scores /= math.sqrt(self.head_dim)

        # Apply mask if causal (must come before softmax)
        if self.causal:
            qk_scores.masked_fill_(self.mask, float("-inf"))

        qk_scores = F.softmax(qk_scores, dim=-1)

        output_with_heads = qk_scores @ v

        output_without_heads = rearrange(output_with_heads, "b h t d -> b t (h d)")

        return self.out_proj(output_without_heads)


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
        linear_module_up=nn.Linear,
        linear_module_down=nn.Linear,
        pre_norm=True,
        normformer=False,
        post_norm=True,
        residual_path=True,
    ):
        super().__init__()

        self.residual_path = residual_path
        self.post_norm = post_norm

        if self.post_norm:
            self.layernorm = nn.LayerNorm(output_features)

        if activation_kwargs is not None:
            self.activation = activation(**activation_kwargs)
        else:
            self.activation = activation()

        self.dropout = nn.Dropout(dropout)

        self.max_features = (
            2 * ratio * output_features
            if activation.__name__.endswith("GLU")
            else ratio * output_features
        )

        self.process = nn.Sequential(
            *[
                nn.LayerNorm(input_features) if pre_norm else nn.Identity(),
                linear_module_up(input_features, self.max_features),
                self.activation,
                nn.LayerNorm(ratio * output_features) if normformer else nn.Identity(),
                linear_module_down(ratio * output_features, output_features),
                self.dropout,
            ]
        )

    def forward(self, x):
        if self.residual_path and self.post_norm:
            return self.layernorm(x + self.process(x))
        elif self.residual_path:
            return x + self.process(x)
        else:
            return self.process(x)


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
        position_embedding_type="absolute",  # absolute or relative
        source_size=None,
        bos_tokens=0,
        mlp_ratio=4,
        activation: nn.Module = nn.ReLU,
        activation_kwargs: Optional[dict] = None,
        ff_linear_module_up=None,
        ff_linear_module_down=None,
        mlp_dropout=0.0,
        msa_dropout=0.0,
        identity_probability=0.0,
        causal=False,
        linear_module=nn.Linear,
        pre_norm=True,
        post_norm=False,
        normformer=False,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.normformer = normformer

        self.drop_path = DropPath(drop_prob=identity_probability, scale_by_keep=True)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)

        if position_embedding_type == "relative":
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
            bos_tokens=bos_tokens,
        )

        # Submodule for the feedforward process
        self.ff = FeedforwardBlock(
            d_model,
            mlp_ratio,
            d_model,
            activation=activation,
            activation_kwargs=activation_kwargs,
            dropout=mlp_dropout,
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
        )

    @property
    def _kv_distance(self) -> float:
        return self.attn._kv_distance

    def forward(self, x):

        if self.pre_norm:
            normx = self.layer_norm_1(x)
            x = x + self.drop_path(self.attn(normx, normx, normx))
            normx = self.layer_norm_2(x)
            x = x + self.drop_path(self.ff(normx))
        elif self.post_norm:
            x = x + self.drop_path(self.attn(x, x, x))
            x = self.layer_norm_1(x)
            x = x + self.drop_path(self.ff(x))
            x = self.layer_norm_2(x)
        else:
            x = x + self.drop_path(self.attn(x, x, x))
            x = x + self.drop_path(self.ff(x))

        if self.pre_norm and self.post_norm:
            x = self.layer_norm_3(x)

        return x


class TransformerEncoder(nn.Module):
    """
    This assumes we already get a sequence of embeddings (e.g. word or image
        patch embeddings). It uses learned positional embeddings.
    """

    def __init__(
        self,
        seq_len,
        d_model,
        n_layers,
        n_heads,
        position_embedding_type="absolute",  # absolute or relative
        source_size=None,
        mlp_ratio=4,
        activation: nn.Module = nn.ReLU,
        activation_kwargs: Optional[dict] = None,
        ff_linear_module_up=None,
        ff_linear_module_down=None,
        mlp_dropout=0.0,
        msa_dropout=0.0,
        stochastic_depth=0.0,
        causal=False,
        linear_module=nn.Linear,
        bos_tokens=0,
        return_bos_tokens=False,
        pre_norm=True,
        post_norm=False,
        normformer=False,
    ):
        if position_embedding_type == "relative":
            assert source_size is not None  # TODO: make this a proper exception

        super().__init__()
        self.seq_len = seq_len
        self.n_heads = n_heads
        self._bos_tokens = bos_tokens
        self.return_bos_tokens = return_bos_tokens

        # Initialise BOS tokens with normal init, like usual Pytorch embeddings
        if self._bos_tokens:
            self._bos_embedding = nn.Parameter(torch.empty(self._bos_tokens, d_model))
            nn.init.normal_(self._bos_embedding, mean=0.0, std=1.0)
            self.full_sequence_length = self.seq_len + self._bos_tokens
        else:
            self._bos_embedding = None
            self.full_sequence_length = self.seq_len

        self.d_model = d_model

        self.position_embedding_type = position_embedding_type

        if self.position_embedding_type == "absolute":
            self.absolute_position_embedding = nn.Embedding(
                self.full_sequence_length, d_model
            )

        self.mlp_dropout = mlp_dropout
        self.msa_dropout = msa_dropout
        self.stochastic_depth = stochastic_depth

        assert isinstance(n_layers, int)  # XXX: make this a proper Exception
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
                    position_embedding_type=position_embedding_type,
                    source_size=source_size,
                    bos_tokens=bos_tokens,
                    mlp_ratio=mlp_ratio,
                    activation=activation,
                    activation_kwargs=activation_kwargs,
                    ff_linear_module_up=ff_linear_module_up,
                    ff_linear_module_down=ff_linear_module_down,
                    mlp_dropout=mlp_dropout,
                    msa_dropout=msa_dropout,
                    identity_probability=self.stochastic_depth_probabilities[i],
                    causal=causal,
                    linear_module=linear_module,
                    pre_norm=pre_norm,
                    post_norm=post_norm,
                    normformer=normformer,
                )
                for i in range(n_layers)
            ]
        )

    @property
    def _kv_distances(self) -> float:
        return ",".join([str(block._kv_distance) for block in self.blocks])

    def forward(self, x):
        if self._bos_tokens:
            x = torch.cat([self._bos_embedding.expand(x.size(0), -1, -1), x], dim=1)
        else:
            x = x

        if self.position_embedding_type == "absolute":
            x = x + self.absolute_position_embedding(
                torch.arange(
                    0, self.full_sequence_length, dtype=torch.long, device=x.device
                ).unsqueeze(
                    0
                )  # to shape (1, seq_len) to broadcast over batch
            )

        for block in self.blocks:
            x = block(x)

        if self._bos_tokens and not self.return_bos_tokens:
            return x[:, self._bos_tokens :, :]
        else:
            return x
