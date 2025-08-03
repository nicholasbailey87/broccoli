"""
Jordan (2024) was able to train a CNN to 94% accuracy on CIFAR-10 in 3.29 seconds
on a single A100 GPU by using carefully-tuned hyperparameters and a number of
techniques to increase learning efficiency. The author notes that applying fixed
weights to the first layer of the network that approximate a whitening
transformation on image patches, following tsyam-code, (2023), was "the single
most impactful feature... [and] more than doubles training speed".

The usefulness of a fixed layer that whitens image patches can be justified
according to the work of Chowers & Weiss (2022), who find that the first layer
weights of a convolutional neural network will asymptotically approach a whitening
transformation regardless of the details of the rest of the network architecture
or the training data. This effectively functions as a bandpass filter layer,
reminiscent of the way neurons in the human primary visual cortex work (Kristensen
& Sandberg, 2021).

The `eigenvectors` function here is adapted from
    https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench96_faster.py
    using https://datascienceplus.com/understanding-the-covariance-matrix/
"""

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
