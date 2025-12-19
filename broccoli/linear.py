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
        xglu=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.xglu = xglu
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

    def reset_rows(self, indices, incoming_data=None):
        """
        Resets rows.
        If incoming_data is provided, resets to the centroid (mean) of that data.
        If not, resets to the mean of existing weights.
        """
        if not torch.is_tensor(indices):
            idx_tensor = torch.as_tensor(
                list(indices), dtype=torch.long, device=self.linear.weight.device
            )
        else:
            idx_tensor = indices

        if idx_tensor.numel() == 0:
            return

        if incoming_data is not None:
            target_center = self._mean_input_weights(incoming_data)
        else:
            target_center = self._mean_value_weights()

        target_center = target_center.expand(idx_tensor.size(0), -1)

        if self.xglu:
            gate_indices = idx_tensor
            value_indices = idx_tensor + (self.linear.out_features // 2)
            self._update_weights(gate_indices, 0, target_center, self.optimisers)
            self._update_weights(value_indices, 0, target_center, self.optimisers)
        else:
            self._update_weights(idx_tensor, 0, target_center, self.optimisers)

    def reset_columns(self, indices):
        if not torch.is_tensor(indices):
            idx_tensor = torch.as_tensor(
                list(indices), dtype=torch.long, device=self.linear.weight.device
            )
        else:
            idx_tensor = indices

        if idx_tensor.size(0):
            random_weights = self._random_weights(
                self.linear.weight.size(0), indices.size(0)
            )
            # Make random col weights quiet so they don't introduce loud noise...
            # ...but not so quiet that FP16 zeros them and ruins symmetry breaking!
            random_weights *= 0.1
            self._update_weights(indices, 1, random_weights, self.optimisers)  # dim
        else:
            return

    def forward(self, x):
        if self.training and self.optimisers:
            self.reset_rows(self.get_reset_indices(0))
            self.reset_columns(self.get_reset_indices(1))
        elif self.training and not self._warned_about_registration:
            warnings.warn(
                "RecyclingLinear: No optimiser registered. Recycling disabled.",
                stacklevel=2,
            )
            self._warned_about_registration = True

        return self.linear(x)

    def get_reset_indices(self, dim):
        base_rate = self.row_recycling_rate if dim == 0 else self.column_recycling_rate
        p = base_rate * self._get_multiplier()
        if dim == 0:
            if self.xglu:
                sample_space = self.linear.out_features // 2
            else:
                sample_space = self.linear.out_features
        elif dim == 1:
            sample_space = self.linear.in_features
        else:
            raise ValueError("`dim` must be 0 or 1")

        # Sample the indices
        probs = torch.rand(sample_space, device=self.linear.weight.device)
        mask = probs < p
        if mask.any():
            return torch.nonzero(mask).squeeze(-1)
        else:
            return torch.tensor([], dtype=torch.long, device=self.linear.weight.device)

    def _random_weights(self, rows, columns):
        device = self.linear.weight.device
        weights = self.linear.weight.data
        stdv = 1.0 / math.sqrt(weights.size(1))
        random_weights = torch.rand(rows, columns, device=device)
        random_weights -= 0.5  # Range [-0.5, +0.5]
        random_weights *= 2.0 * stdv  # Range [-stdv, +stdv]
        return random_weights

    def _mean_input_weights(self, input):
        reduce_dims = list(range(input.ndim - 1))
        data_mean = input.detach().mean(dim=reduce_dims, keepdim=True)

        weights = self.linear.weight.data
        stdv = 1.0 / math.sqrt(weights.size(1))
        data_norm = data_mean.std() + 1e-6
        scale_factor = stdv / data_norm

        return data_mean * scale_factor

    def _mean_value_weights(self):
        """
        Only used when self.xglu
        """
        weights = self.linear.weight.data
        rows = weights.size(0)
        if self.xglu:
            return self.linear.weight[int(rows / 2) :].data.mean(dim=0, keepdim=True)
        else:
            return self.linear.weight.data.mean(dim=0, keepdim=True)

    def _mean_gate_weights(self):
        """
        Only used when self.xglu
        """
        weights = self.linear.weight.data
        rows = weights.size(0)
        return self.linear.weight[: int(rows / 2)].data.mean(dim=0, keepdim=True)

    def _update_weights(
        self,
        indices: Iterable[int],
        dim: int,
        data: torch.Tensor,
        optimisers: Union[
            List[torch.optim.Optimizer], torch.optim.Optimizer, None
        ] = None,
    ):
        if optimisers is None:
            optimisers = []
        if not isinstance(optimisers, list):
            optimisers = [optimisers]

        if not torch.is_tensor(indices):
            idx_tensor = torch.as_tensor(
                list(indices), dtype=torch.long, device=self.linear.weight.device
            )
        else:
            idx_tensor = indices

        if idx_tensor.numel() == 0:
            return

        with torch.no_grad():
            if dim == 0:
                self.linear.weight.data[idx_tensor] = data
            elif dim == 1:
                self.linear.weight.data[:, idx_tensor] = data
            else:
                raise ValueError("`dim` must be 0 or 1")

            self._reset_optim_state(self.linear.weight, idx_tensor, optimisers, dim=dim)

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
