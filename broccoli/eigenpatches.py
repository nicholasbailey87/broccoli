import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from einops import rearrange


def eigenvectors(images: torch.Tensor, patch_size: int = 2, eps=5e-4) -> torch.Tensor:
    """
    Adapted from
        https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench96_faster.py
        using https://datascienceplus.com/understanding-the-covariance-matrix/
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


def get_eigenpatches(data: DataLoader):
    patches = None
    for images, _ in data:
        eigenpatches = eigenvectors(images)
        if patches is None:
            patches = eigenpatches
        else:
            patches = 0.99 * patches + 0.01 * eigenpatches
    return patches


def plottable_bandpass_filters(eigenpatches: torch.Tensor, h, w, c):
    bandpass_filters = rearrange(eigenpatches, "N (C H W) -> N H W C", C=3, H=2, W=2)
    return bandpass_filters.detach().numpy()
