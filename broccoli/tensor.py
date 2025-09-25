import torch
from torch import nn
from torch.nn import functional as F


class SigmaReparamTensor(nn.Module):
    """
    Inspired by Apple's Spectral Normed Linear Layers
        (https://github.com/apple/ml-sigma-reparam)
    """

    def __init__(self, init_tensor: torch.Tensor):
        assert init_tensor.ndim == 2

        super().__init__()

        self.sigma_reparam_tensor = nn.Parameter(init_tensor, requires_grad=True)

        with torch.no_grad():
            _, sigma, v_transpose = torch.linalg.svd(
                self.sigma_reparam_tensor, full_matrices=False
            )

        self.register_buffer("approx_spectral_norm", sigma[:1])
        self.register_buffer("right_singular", v_transpose[0])
        self.sigma_reparam_scale = nn.Parameter(
            self.approx_spectral_norm.clone().detach(), requires_grad=True
        )

    def power_iteration(self):
        with torch.no_grad():
            approx_right_singular_transpose = self.sigma_reparam_tensor.mv(
                self.right_singular
            )
            approx_right_singular_transpose = F.normalize(
                approx_right_singular_transpose, dim=0
            )
            updated_right_singular = self.sigma_reparam_tensor.T.mv(
                approx_right_singular_transpose
            )
            updated_right_singular = F.normalize(updated_right_singular, dim=0)
            self.right_singular.data.copy_(updated_right_singular)
            rayleigh_quotient = torch.einsum(
                "m,mn,n->",
                approx_right_singular_transpose,
                self.sigma_reparam_tensor,
                updated_right_singular,
            )
            self.approx_spectral_norm.data.copy_(rayleigh_quotient)

    def forward(self):
        if self.training:
            self.power_iteration()
        return self.sigma_reparam_scale * (
            self.sigma_reparam_tensor / self.approx_spectral_norm
        )


class AnchoredReparamTensor(nn.Module):
    """
    Reparameterises a tensor by decoupling its magnitude and direction.

    The direction is represented by a learnable weight tensor, normalised by the
    Rayleigh quotient with respect to its initial dominant right-singular vector.
    The magnitude is a separate learnable scalar.

    The reparameterization is:

        W_reparam = scale * (W / norm)

    where the norm is the Rayleigh quotient uᵀWv₀, with v₀ being the dominant
    right-singular vector of the initial tensor and u = normalize(Wv₀).
    """

    def __init__(self, init_tensor: torch.Tensor):
        assert init_tensor.ndim == 2

        super().__init__()

        self.weight = nn.Parameter(init_tensor, requires_grad=True)

        with torch.no_grad():
            _, sigma, v_transpose = torch.linalg.svd(self.weight, full_matrices=False)

        self.register_buffer("rayleigh_norm", sigma[:1])
        self.register_buffer("initial_right_singular", v_transpose[0])
        self.nondecay_scale = nn.Parameter(
            sigma[:1].clone().detach(), requires_grad=True
        )

    def _update_rayleigh_norm(self):
        with torch.no_grad():
            product = self.weight.mv(self.initial_right_singular)
            normed_product = F.normalize(product, dim=0)
            rayleigh_norm = torch.einsum(
                "m,mn,n->",
                normed_product,
                self.weight,
                self.initial_right_singular,
            )
            self.rayleigh_norm.data.copy_(rayleigh_norm)

    def forward(self):
        if self.training:
            self._update_rayleigh_norm()
        return self.nondecay_scale * (self.weight / (self.rayleigh_norm + 1e-6))


class NormReparamTensor(nn.Module):
    """
    Reparameterise a tensor as a normalised tensor of weights multiplied by a
        learnable scaling factor.
    """

    def __init__(self, init_tensor: torch.Tensor):
        assert init_tensor.ndim == 2, "Input tensor must be a 2D matrix."
        super().__init__()

        # Use the gradboard convention of calling something nondecay_* if we should
        # exclude it from weight decay
        self.weight = nn.Parameter(init_tensor.clone(), requires_grad=True)
        self.nondecay_scale = nn.Parameter(
            torch.linalg.norm(self.weight).clone().detach(), requires_grad=True
        )

    def forward(self) -> torch.Tensor:
        norm = torch.linalg.norm(self.weight)
        return self.nondecay_scale * (self.weight / (norm + 1e-6))
