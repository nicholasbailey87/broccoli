import torch
from torch import nn
from torch.nn import functional as F


class ReLU(nn.Module):
    """
    A ReLU activation function with optional clamp and leakiness.
    """

    def __init__(
        self, clamp=True, leaky=True, negative_slope=0.01, clamp_max=6.0
    ) -> None:
        super().__init__()
        self.clamp = clamp
        self.leaky = leaky
        self.negative_slope = negative_slope
        self.clamp_max = clamp_max

    def forward(self, x):
        if self.leaky:
            relu = F.leaky_relu(x, negative_slope=self.negative_slope)
        else:
            relu = F.relu(x)
        if self.clamp:
            relu = torch.clamp(relu, max=self.clamp_max)
        return relu


class GELU(nn.Module):
    """
    A GELU activation function with optional clamp.
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


class Swish(nn.Module):
    """
    Implementation of (beta) SwiGLU, as introduced in "GLU Variants Improve Transformer"
        (https://arxiv.org/abs/2002.05202v1) and used to great effect in LLaMa 2.0.

    Halves the incoming parameter count, which should be scaled up before input.
    """

    def __init__(self) -> None:
        super().__init__()
        # Learnable parameter is called "swiglu beta" so that it is easy to find
        #   and exclude from weight decay
        self.swish_beta = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return x * F.sigmoid(self.swish_beta * x)


class SquaredReLU(nn.Module):
    """
    Squared ReLU, as shown in "ReLU^2 wins" (https://arxiv.org/abs/2402.03804) to
      be as effective as SwiGLU for training LLMs, possibly because it can allow a
      NN to learn multyiplication, as noted by
      https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
    """

    def __init__(
        self, clamp=True, leaky=True, negative_slope: float = 0.01, clamp_max=6
    ) -> None:
        super().__init__()
        self.clamp = clamp
        self.leaky = leaky
        self.negative_slope = negative_slope
        self.clamp_max = clamp_max

    def forward(self, x):
        if self.leaky:
            relu = F.leaky_relu(x, negative_slope=self.negative_slope)
        else:
            relu = F.relu(x)
        relu_squared = relu**2
        if self.clamp:
            relu_squared = torch.clamp(relu_squared, max=self.clamp_max)
        return relu_squared


class XGLU(nn.Module):
    """
    Generic Gated Linear Unit
    """

    def __init__(self, activation_module: nn.Module) -> None:
        super().__init__()
        self.activation = activation_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = x.chunk(2, dim=-1)
        return self.activation(gate) * value


def SquaredReGLU(clamp=True, leaky=True, negative_slope=0.01, clamp_max=6.0) -> XGLU:
    """
    Factory function that creates a GLU with a SquaredReLU activation.
    """
    activation_module = SquaredReLU(
        clamp=clamp, leaky=leaky, negative_slope=negative_slope, clamp_max=clamp_max
    )
    return XGLU(activation_module)


def SwiGLU() -> XGLU:
    """
    Factory function that creates a GLU with a Swish activation.
    """
    return XGLU(Swish())
