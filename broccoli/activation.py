import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class SwiGLU(nn.Module):
    """
    Implementation of (beta) SwiGLU, as introduced in "GLU Variants Improve Transformer"
        (https://arxiv.org/abs/2002.05202v1) and used to great effect in LLaMa 2.0.

    Halves the incoming parameter count, which should be scaled up before input.
    """

    def __init__(self, linear_module: nn.Module = nn.Linear) -> None:
        super().__init__()
        # Learnable parameter is called "swiglu beta" so that it is easy to find
        #   and exclude from weight decay
        self.swiglu_beta = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        gate, value = rearrange(x, "... (split c) -> split ... c", split=2)
        beta_swish = gate * F.sigmoid(self.swiglu_beta * gate)
        return beta_swish * value
