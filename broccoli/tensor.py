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

        self.tensor = nn.Parameter(init_tensor, requires_grad=True)

        with torch.no_grad():
            _, sigma, v_transpose = torch.linalg.svd(self.tensor, full_matrices=False)

        self.register_buffer("approx_spectral_norm", sigma[:1])
        self.register_buffer("right_singular", v_transpose[0])
        self.scale = nn.Parameter(
            self.approx_spectral_norm.clone().detach(), requires_grad=True
        )

    def power_iteration(self):
        with torch.no_grad():
            approx_right_singular_transpose = self.tensor.mv(self.right_singular)
            approx_right_singular_transpose = F.normalize(
                approx_right_singular_transpose, dim=0
            )
            updated_right_singular = self.tensor.T.mv(approx_right_singular_transpose)
            updated_right_singular = F.normalize(self.right_singular, dim=0)
            self.right_singular.data.copy_(updated_right_singular)
            rayleigh_quotient = torch.einsum(
                "m,mn,n->",
                approx_right_singular_transpose,
                self.tensor,
                updated_right_singular,
            )
            self.approx_spectral_norm.data.copy_(rayleigh_quotient)

    def forward(self):
        if self.training:
            self.power_iteration()
        return self.scale * (self.tensor / self.approx_spectral_norm)
