import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union

from einops.layers.torch import Rearrange


def spatial_tuple(size: Union[int, tuple], spatial_dimensions):
    """
    Converts an integer x to `tuple([x] * spatial_dimensions)`.
    Performs no operation (i.e. the identity operation) on tuples of length `spatial_dimensions`.
    Otherwise
    """
    if isinstance(size, int):
        return tuple([size] * spatial_dimensions)
    elif isinstance(size, tuple) and (len(size) == spatial_dimensions):
        return size
    else:
        raise ValueError(
            f"For {spatial_dimensions} spatial dimensions, `size` must be "
            f"an integer or a tuple of length {spatial_dimensions}."
        )


def padding_tensor(padding: tuple):
    """
    Converts a tuple of ints (x, y, z) into a tuple of 2-tuples,
        like ((x, x), (y, y), (z, z)).

    Performs no operation (i.e. the identity operation) on a tuple of 2-tuples.

    Otherwise raises an error.
    """
    if all(isinstance(x, int) for x in padding):
        return tuple([tuple([p] * 2) for p in padding])
    elif (
        all(isinstance(p, tuple) for p in padding)
        and all(len(p) == 2 for p in padding)
        and all(all(isinstance(x, int) for x in p) for p in padding)
    ):
        return padding
    else:
        raise ValueError(
            "Padding must be a tuple of ints of a tuple of 2-tuples of ints. "
            f"It was {padding}."
        )


def kd_unfold(t: torch.Tensor, kernel_size=1, stride=1, padding=0, k=2):
    """
    Unfold operation with k spatial dimensions.
    Does not support dilation.
    Only supports equal padding at top and bottom.
    """
    if len(t.size()[2:]) != k:
        raise ValueError(
            f"Input tensor size should be (N, channels, spatial dims...), so "
            f"for k = {k}, t.size() should be a tuple of length {k + 2}."
        )

    N, C = t.size(0), t.size(1)

    kernel_size = spatial_tuple(kernel_size, k)
    stride = spatial_tuple(stride, k)
    padding = padding_tensor(spatial_tuple(padding, k))

    output = t
    output = F.pad(output, sum(reversed(padding), ()))  # i.e. the empty tuple

    for i, _ in enumerate(kernel_size):
        output = output.unfold(i + 2, kernel_size[i], stride[i])

    permutation = [0, 1] + [i + k + 2 for i in range(k)] + [i + 2 for i in range(k)]

    return output.permute(*permutation).reshape(N, math.prod(kernel_size) * C, -1)


def calculate_output_spatial_size(
    input_spatial_size, kernel_size=1, stride=1, padding=0, dilation=0
):
    """
    Calculate the output size for the spatial dimensions of a convolutional operation
    """
    stride = spatial_tuple(stride, len(input_spatial_size))

    # Handle padding keywords that are sometimes used
    if padding == "same":
        output_size = ()
        for i, in_length in enumerate(input_spatial_size):
            output_size += (math.ceil(in_length / stride[i]),)
        return output_size
    elif padding == "valid":
        padding = 0

    kernel_size = spatial_tuple(kernel_size, len(input_spatial_size))
    padding = spatial_tuple(padding, len(input_spatial_size))
    dilation = spatial_tuple(dilation, len(input_spatial_size))

    output_size = ()

    for i, in_length in enumerate(input_spatial_size):
        output_size += (
            math.floor(
                (in_length + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1)
                / stride[i]
                + 1
            ),
        )
    return output_size


class SpaceToDepth(nn.Module):
    """
    An operation that extracts patches from an image-like tensor and stacks
        them channel-wise.
    """

    def __init__(self, kernel_size, stride=1, padding=0, spatial_dimensions=2):
        """
        Input shape should be in order (channels, spatial dims...),
            e.g. (channels, height, width)
        """

        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.spatial_dimensions = spatial_dimensions

    def forward(self, x):

        N, C, *input_spatial_size = x.size()

        patches = kd_unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            k=self.spatial_dimensions,
        )

        output_spatial_size = calculate_output_spatial_size(
            input_spatial_size=input_spatial_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=1,  # kd_unfold doesn't support dilation
        )

        output_channels = C * math.prod(
            spatial_tuple(self.kernel_size, self.spatial_dimensions)
        )

        return patches.view(N, output_channels, *output_spatial_size)
