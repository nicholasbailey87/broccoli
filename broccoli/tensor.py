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
    Reparameterise a tensor as a normalised tensor of weights multiplied by a
        learnable scaling factor.

    The tensor of weights is also reparameterised as the product of a learnable
        weight tensor with the (fixed) dominant right-singular vector of the
        weight tensor as it was initialised.

    i.e this module represents a tensor reparameterised as:

        W_reparam = scale * (W / ||W @ v_0||_2)

        where v_0 is the dominant right-singular vector of the initial tensor W_init.
    """

    def __init__(self, init_tensor: torch.Tensor):
        assert init_tensor.ndim == 2, "Input tensor must be a 2D matrix."
        super().__init__()

        # Use the gradboard convention of calling something nondecay_* if we should
        # exclude it from weight decay
        self.nondecay_weight = nn.Parameter(init_tensor.clone(), requires_grad=True)

        # At initialization, compute the dominant right-singular vector (v_0)
        # and store it in a non-trainable buffer.
        with torch.no_grad():
            _, _, v_transpose = torch.linalg.svd(
                self.nondecay_weight, full_matrices=False
            )
            # v_transpose[0] is the first row of V^T, which is the first right-singular vector.
            self.register_buffer("anchor_vector", v_transpose[0])

        initial_norm = torch.linalg.vector_norm(
            self.nondecay_weight.mv(self.anchor_vector)
        )
        self.scale = nn.Parameter(initial_norm.clone().detach(), requires_grad=True)

    def forward(self) -> torch.Tensor:
        # Calculate the L2 norm of the matrix-vector product W @ v_0
        norm = torch.linalg.vector_norm(self.nondecay_weight.mv(self.anchor_vector))

        # Return the reparameterized tensor.
        return self.scale * (self.nondecay_weight / (norm + 1e-6))


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
        self.nondecay_weight = nn.Parameter(init_tensor.clone(), requires_grad=True)
        self.scale = nn.Parameter(
            torch.linalg.norm(self.nondecay_weight).clone().detach(), requires_grad=True
        )

    def forward(self) -> torch.Tensor:
        return self.scale * F.normalize(self.nondecay_weight)
