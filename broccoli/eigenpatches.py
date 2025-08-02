import torch
import torch.nn as nn
from einops import rearrange


def eigenvectors(images: torch.Tensor, patch_size: int, eps=5e-4) -> torch.Tensor:
    """
    Adapted from
        github.com/KellerJordan/cifar10-airbench/blob/master/airbench96_faster.py
        using https://datascienceplus.com/understanding-the-covariance-matrix/

    Args:
        images: a batch of training images (the bigger and more representative the better!)
        patch_size: the size of the eigenvectors we want to create (i.e. the patch/kernel
            size of the model we will initialise with the eigenvectors)
        eps: a small number to avoid division by zero
    """
    with torch.no_grad():
        unfolder = nn.Unfold(kernel_size=patch_size, stride=1)
        patches = unfolder(images)  # (N, patch_elements, patches_per_image)
        patches = rearrange(patches, "N elements patches -> (N patches) elements")
        n = patches.size(0)
        centred = patches - patches.mean(dim=1, keepdim=True)
        covariance_matrix = (
            centred.T @ centred
        ) / n  # https://datascienceplus.com/understanding-the-covariance-matrix/
        _, eigenvectors = torch.linalg.eigh(covariance_matrix)
        return eigenvectors
