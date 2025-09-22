# UNDER CONSTRUCTION

import math
import torch
from torch import nn
from torch.nn import functional as F

from .tensor import SigmaReparamTensor, AnchoredReparamTensor, NormReparamTensor


class SpectralNormLinear(nn.Module):
    """
    Inspired by Apple's Spectral Normed Linear Layers
        (https://github.com/apple/ml-sigma-reparam)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weights = None

        # Define the bias vector as a learnable parameter if required.
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            # If no bias, register it as None.
            # This is important so that PyTorch doesn't complain when saving/loading the model.
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.weights = SigmaReparamTensor(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weights(), self.bias)

    def __repr__(self) -> str:
        # Optional: A nice representation for printing the module.
        return (
            f"SpectralNormFeedForward(in_features={self.in_features},"
            f"out_features={self.out_features}, bias={self.use_bias})"
        )


class AnchoredLinear(nn.Module):
    """
    ...
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weights = None

        # Define the bias vector as a learnable parameter if required.
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            # If no bias, register it as None.
            # This is important so that PyTorch doesn't complain when saving/loading the model.
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.weights = AnchoredReparamTensor(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weights(), self.bias)

    def __repr__(self) -> str:
        # Optional: A nice representation for printing the module.
        return (
            f"AnchoredLinear(in_features={self.in_features},"
            f"out_features={self.out_features}, bias={self.use_bias})"
        )


class WeightNormedLinear(nn.Module):
    """
    ...
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weights = None

        # Define the bias vector as a learnable parameter if required.
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            # If no bias, register it as None.
            # This is important so that PyTorch doesn't complain when saving/loading the model.
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.weights = NormReparamTensor(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weights(), self.bias)

    def __repr__(self) -> str:
        # Optional: A nice representation for printing the module.
        return (
            f"AnchoredLinear(in_features={self.in_features},"
            f"out_features={self.out_features}, bias={self.use_bias})"
        )
