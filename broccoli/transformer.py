import math
from collections import OrderedDict
from typing import Optional
from numpy import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, einsum, reduce, repeat
from einops.layers.torch import Rearrange


class MHAttention(nn.Module):
    """
    Multi-head self-attention using einops and custom linear layer.

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
        sequence_length=None,
        share_kv=True,
        linear_module: nn.Module = nn.Linear,
    ):
        super().__init__()
        if causal:
            assert sequence_length is not None
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0
        self.head_dim = self.embed_dim // self.n_heads
        self.share_kv = share_kv
        self.q_proj = linear_module(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = linear_module(self.embed_dim, self.embed_dim, bias=False)
        if self.share_kv:
            self.v_proj = self.k_proj
        else:
            self.v_proj = linear_module(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = linear_module(self.embed_dim, self.embed_dim, bias=False)
        self.causal = causal
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)
        if self.causal:
            self.register_buffer(
                "mask",
                (
                    torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1)
                    == 1
                )
                .unsqueeze(0)
                .unsqueeze(0),
            )

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

        # Project q, k and v and divide into heads
        q = rearrange(self.q_proj(q), "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(self.k_proj(k), "b t (h d) -> b h t d", h=self.n_heads)
        if self.share_kv:
            v = k
        else:
            v = rearrange(self.v_proj(v), "b t (h d) -> b h t d", h=self.n_heads)

        qk_scores = q @ k.transpose(-1, -2)
        qk_scores /= math.sqrt(self.head_dim)  # scaling

        # Apply mask if causal (must come before softmax)
        if self.causal:
            qk_scores.masked_fill_(self.mask, float("-inf"))

        qk_scores = torch.softmax(qk_scores, dim=-1)  # softmax
        qk_scores = self.dropout(qk_scores)  # dropout must come after softmax!

        output_with_heads = qk_scores @ v

        output_without_heads = rearrange(output_with_heads, "b h t d -> b t (h d)")

        return self.out_proj(output_without_heads)


class TransformerBlock(nn.Module):
    """
    Performs LayerNorms first (as in PyTorch Transformers when norm_first=True),
        which is also what is seen in e.g.
        https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        and is recommended by https://arxiv.org/abs/2002.04745

    """

    def __init__(
        self,
        d_model,
        n_heads,
        mlp_ratio=4,
        activation: nn.Module = nn.ReLU,
        activation_kwargs: Optional[dict] = None,
        mlp_dropout=0.0,
        msa_dropout=0.0,
        causal=False,
        linear_module=nn.Linear,
    ):
        super().__init__()

        if activation_kwargs is not None:
            self.activation = activation(**activation_kwargs)
        else:
            self.activation = activation()

        # Submodules for applying attention
        self.layer_norm = nn.LayerNorm(d_model)
        self.attn = MHAttention(  # Handles QKV projection
            d_model,
            n_heads,
            dropout=msa_dropout,
            causal=causal,
            linear_module=linear_module,
        )

        # Submodules for the feedforward process
        self.ff_process = nn.Sequential(
            OrderedDict(
                [
                    ("layer_norm", nn.LayerNorm(d_model)),
                    (
                        # up_projection is appropriate to activation
                        "up_projection",
                        linear_module(
                            d_model,
                            (
                                2 * mlp_ratio * d_model
                                if activation.__name__.endswith("GLU")
                                else mlp_ratio * d_model
                            ),
                        ),
                    ),
                    # xGLU activations will halve embedding size
                    ("activation", self.activation),
                    ("down_projection", linear_module(mlp_ratio * d_model, d_model)),
                    ("dropout", nn.Dropout(mlp_dropout)),
                ]
            )
        )

    def forward(self, x):
        normx = self.layer_norm(x)
        x = x + self.attn(normx, normx, normx)
        x = x + self.ff_process(x)
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
        mlp_ratio=4,
        activation: nn.Module = nn.ReLU,
        activation_kwargs: Optional[dict] = None,
        mlp_dropout=0.0,
        msa_dropout=0.0,
        stochastic_depth=0.0,
        causal=False,
        linear_module=nn.Linear,
        bos_tokens=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_heads = n_heads

        # Initialise BOS tokens with Xavier uniform init per
        # https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
        if bos_tokens:
            self.bos_tokens = nn.Parameter(torch.empty(bos_tokens, d_model))
            nn.init.normal_(self.bos_tokens, mean=0.0, std=1.0)
            self.full_sequence_length = self.seq_len + bos_tokens
        else:
            self.bos_tokens = None
            self.full_sequence_length = self.seq_len

        self.d_model = d_model
        self.positional_embedding = nn.Embedding(self.full_sequence_length, d_model)
        self.mlp_dropout = mlp_dropout
        self.msa_dropout = msa_dropout
        self.stochastic_depth = stochastic_depth
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    mlp_ratio=mlp_ratio,
                    activation=activation,
                    activation_kwargs=activation_kwargs,
                    mlp_dropout=mlp_dropout,
                    msa_dropout=msa_dropout,
                    causal=causal,
                    linear_module=linear_module,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        if self.bos_tokens is not None:
            x = torch.cat([self.bos_tokens.expand(x.size(0), -1, -1), x], dim=1)
        else:
            x = x
        x = x + self.positional_embedding(
            torch.arange(
                0, self.full_sequence_length, dtype=torch.long, device=x.device
            ).unsqueeze(
                0
            )  # to shape (1, seq_len) to broadcast over batch
        )
        for block in self.blocks:
            if (not self.training) or self.stochastic_depth == 0.0:
                x = block(x)
            else:  # drop out some rows from the next Transformer block operation
                binomial = random.binomial(n=x.size(0), p=1 - self.stochastic_depth)
                shuffle_indices = torch.randperm(x.size(0), device=x.device)
                unshuffle_indices = torch.argsort(shuffle_indices)  # , device=x.device)
                shuffled = x[shuffle_indices, :, :]
                include = shuffled[:binomial, :, :]
                exclude = shuffled[binomial:, :, :]
                x = torch.cat([block(include), exclude])[
                    unshuffle_indices, :, :
                ].contiguous()

        if self.bos_tokens is not None:
            return x[:, (self.full_sequence_length - self.seq_len) :, :]
        else:
            return x
