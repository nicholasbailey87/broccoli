# UNDER CONSTRUCTION

import torch
from torch import nn
from torch.nn import functional as F


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
