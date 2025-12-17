import math
import random
import warnings
from typing import Union, List, Iterable

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
        stdv = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(weights, a=-stdv, b=stdv)
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
        stdv = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(weights, a=-stdv, b=stdv)
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
        stdv = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(weights, a=-stdv, b=stdv)
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.weights = NormReparamTensor(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weights(), self.bias)

    def __repr__(self) -> str:
        return (
            f"WeightNormedLinear(in_features={self.in_features},"
            f"out_features={self.out_features}, bias={self.use_bias})"
        )


class RecyclingLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        row_recycling_rate: float = 0.0,
        column_recycling_rate: float = 0.0,
        adaptive=False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.row_recycling_rate = row_recycling_rate
        self.column_recycling_rate = column_recycling_rate
        self.adaptive = adaptive
        self.optimisers = []
        self.initial_learning_rates = []
        self._warned_about_registration = False

    def register_optimiser(self, optimiser: torch.optim.Optimizer):
        self.optimisers.append(optimiser)
        self.initial_learning_rates.append(self._get_learning_rate(optimiser))
        if self.initial_learning_rates[-1] == 0.0:
            warnings.warn(
                "Learning rate of registered optimiser was 0.0 - make sure "
                "you haven't initialised a scheduler before registering the "
                "optimiser",
                stacklevel=2,
            )

    def _get_learning_rate(self, optimiser: torch.optim.Optimizer):
        for group in optimiser.param_groups:
            for param in group["params"]:
                if param is self.linear.weight:
                    return group["lr"]

    def _get_multiplier(self):
        if not self.adaptive or not self.optimisers:
            return 1.0
        else:
            init = self.initial_learning_rates
            current = [self._get_learning_rate(o) for o in self.optimisers]
            pairs = zip(current, init, strict=True)
            multipliers = [a / b for a, b in pairs if b != 0.0]
            return min(multipliers) if multipliers else 0.0

    def forward(self, x):
        multiplier = self._get_multiplier()
        col_recycling_rate = self.column_recycling_rate * multiplier
        row_recycling_rate = self.row_recycling_rate * multiplier

        if self.training and self.optimisers:

            if row_recycling_rate > 0:
                probs = torch.rand(self.linear.out_features, device=x.device)
                mask = probs < row_recycling_rate
                if mask.any():
                    # nonzero returns [N, 1], squeeze to get [N]
                    indices = torch.nonzero(mask).squeeze(-1)
                    self.reset_rows(indices, self.optimisers)

            if col_recycling_rate > 0:
                probs = torch.rand(self.linear.in_features, device=x.device)
                mask = probs < col_recycling_rate
                if mask.any():
                    indices = torch.nonzero(mask).squeeze(-1)
                    self.reset_columns(indices, self.optimisers)

        elif self.training and not self._warned_about_registration:
            warnings.warn(
                "RecyclingLinear: No optimiser registered. Recycling disabled.",
                stacklevel=2,
            )
            self._warned_about_registration = True

        return self.linear(x)

    def reset_rows(
        self,
        indices: Iterable[int],
        optimisers: Union[
            List[torch.optim.Optimizer], torch.optim.Optimizer, None
        ] = None,
    ):
        """
        Update some of the weight rows to be equal to the mean of all weight rows.
        """
        if optimisers is None:
            optimisers = []
        if not isinstance(optimisers, list):
            optimisers = [optimisers]

        device = self.linear.weight.device
        idx_tensor = torch.as_tensor(list(indices), dtype=torch.long, device=device)

        if idx_tensor.numel() == 0:
            return

        with torch.no_grad():
            # Calculate mean of all rows including the rows to be reset
            mean_vector = self.linear.weight.data.mean(
                dim=0, keepdim=True
            )  # [1, in_features]
            update_data = mean_vector.expand(idx_tensor.size(0), -1)
            self.linear.weight.data[idx_tensor] = update_data

            if self.linear.bias is not None:
                self.linear.bias.data[idx_tensor] = 0.0

            self._reset_optim_state(self.linear.weight, idx_tensor, optimisers, dim=0)
            if self.linear.bias is not None:
                self._reset_optim_state(self.linear.bias, idx_tensor, optimisers, dim=0)

    def reset_columns(
        self,
        indices: Iterable[int],
        optimisers: Union[
            List[torch.optim.Optimizer], torch.optim.Optimizer, None
        ] = None,
    ):
        """
        Update some of the weight columns to be random as though reinitialised.
        """
        if optimisers is None:
            optimisers = []
        if not isinstance(optimisers, list):
            optimisers = [optimisers]

        device = self.linear.weight.device
        idx_tensor = torch.as_tensor(list(indices), dtype=torch.long, device=device)

        if idx_tensor.numel() == 0:
            return

        with torch.no_grad():
            # 1. Generate Random Columns
            # Shape: [out_features, N_indices]
            weights = self.linear.weight.data
            stdv = 1.0 / math.sqrt(weights.size(1))

            # Generate [Rows, N] block
            random_weights = torch.rand(
                weights.size(0), idx_tensor.size(0), device=device
            )
            random_weights = (random_weights - 0.5) * 2.0 * stdv

            # 2. Update Weights (One-shot)
            # We assign into the columns specified by idx_tensor
            self.linear.weight.data[:, idx_tensor] = random_weights

            # 3. Update Optimizers
            # Bias is untouched by column resets (bias is shape [Out], cols are [In])
            self._reset_optim_state(self.linear.weight, idx_tensor, optimisers, dim=1)

    def _reset_optim_state(self, param, idx_tensor, optimisers, dim):
        """
        Zeroes out the optimizer state for the given indices in a single operation.
        """
        for optimiser in optimisers:
            if param not in optimiser.state:
                continue
            state = optimiser.state[param]

            for _, buffer in state.items():
                if torch.is_tensor(buffer) and buffer.shape == param.shape:
                    # Vectorized zeroing
                    if dim == 0:
                        buffer[idx_tensor] = 0.0
                    else:
                        buffer[:, idx_tensor] = 0.0
