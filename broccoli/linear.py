# UNDER CONSTRUCTION

import math
import torch
from torch import nn
from torch.nn import functional as F

from .tensor import SigmaReparamTensor


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

        self.weight_init = nn.Parameter(torch.empty(out_features, in_features))

        # Define the bias vector as a learnable parameter if required.
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            # If no bias, register it as None.
            # This is important so that PyTorch doesn't complain when saving/loading the model.
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_init, a=math.sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_init)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.weights = SigmaReparamTensor(self.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weights(), self.bias)

    def __repr__(self) -> str:
        # Optional: A nice representation for printing the module.
        return (
            f"SpectralNormFeedForward(in_features={self.in_features}",
            f"out_features={self.out_features}, bias={self.use_bias})",
        )


class RandomLinear(nn.Linear):
    """ """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,  # <---- TODO: explain this
        beta=0.1,
        forward_looks_random=True,
    ):
        super().__init__(in_features, out_features, bias=False)
        self.beta = beta
        self.forward_looks_random = forward_looks_random

    def forward(self, inputs: torch.Tensor):
        if not self.training:
            return F.linear(inputs, self.weight)
        else:
            # Initialise self.random_weights
            random_weights = torch.empty_like(self.weight)
            nn.init.trunc_normal_(random_weights)
            random_weights *= self.beta

            if self.forward_looks_random:
                # Forward using a reparameterisation trick
                a = F.linear(inputs.detach(), self.weight, self.bias)
                b = F.linear(inputs, random_weights, bias=None)
            else:
                # Forward as (W_actual * input + W_random * input) + bias
                a = F.linear(inputs, self.weight, self.bias)
                b = F.linear(inputs, random_weights, bias=None)

            return a + b
