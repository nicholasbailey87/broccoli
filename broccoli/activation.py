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

    def __init__(self) -> None:
        super().__init__()
        # Learnable parameter is called "swiglu beta" so that it is easy to find
        #   and exclude from weight decay
        self.swiglu_beta = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        gate, value = rearrange(x, "... (split c) -> split ... c", split=2)
        beta_swish = gate * F.sigmoid(self.swiglu_beta * gate)
        return beta_swish * value


class SquaredReLU(nn.Module):
    """
    Squared ReLU, as shown in "ReLU^2 wins" (https://arxiv.org/abs/2402.03804) to
      be as effective as SwiGLU for training LLMs, possibly because it can allow a
      NN to learn multyiplication, as noted by
      https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
    """

    def __init__(self, clamp=True, leaky=True) -> None:
        super().__init__()
        self.clamp = clamp
        self.leaky = leaky

    def forward(self, x):
        if self.leaky:
            relu = F.leaky_relu(x)
        else:
            relu = F.relu(x)
        relu_squared = relu**2
        if self.clamp:
            relu_squared = torch.clamp(relu_squared, max=6)
        return relu_squared


class ReLU(nn.Module):
    """
    A ReLU activation function with optional clamp and leakiness.
    """

    def __init__(self, clamp=True, leaky=True) -> None:
        super().__init__()
        self.clamp = clamp
        self.leaky = leaky

    def forward(self, x):
        if self.leaky:
            relu = F.leaky_relu(x)
        else:
            relu = F.relu(x)
        if self.clamp:
            relu = torch.clamp(relu, max=6)
        return relu


class GELU(nn.Module):
    """
    A ReLU activation function with optional clamp and leakiness.
    """

    def __init__(self, clamp=True) -> None:
        super().__init__()
        self.clamp = clamp
        self.gelu = nn.GELU()

    def forward(self, x):
        gelu = self.gelu(x)
        if self.clamp:
            gelu = torch.clamp(gelu, max=6)
        return gelu
