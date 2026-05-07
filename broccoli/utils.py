import math
import torch.nn as nn
import torch.nn.functional as F
from .cnn import spatial_tuple


class PadTensor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        if sum(self.args[0]) == 0:
            return x
        else:
            return F.pad(x, *self.args, **self.kwargs)


class ResizeAndPadPatches(nn.Module):
    """
    Resizes and pads image patches so that the flattened size will be
        smaller than d_model
    """

    def __init__(self, in_channels, d_model, patch_size, spatial_dimensions=2):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.patch_size = spatial_tuple(patch_size, spatial_dimensions)
        self.spatial_dimensions = spatial_dimensions

        # Figure out the largest downscaled patch volume that fits in d_model
        max_spatial_vol = d_model // in_channels

        if max_spatial_vol == 0:
            raise ValueError(f"patch channels ({in_channels}) > d_model ({d_model}).")

        if spatial_dimensions == 1:
            self.out_size = (max_spatial_vol,)
            self.mode = "linear"
        elif spatial_dimensions == 2:
            side = int(math.sqrt(max_spatial_vol))
            self.out_size = (side, side)
            self.mode = "bilinear"
        elif spatial_dimensions == 3:
            side = int(max_spatial_vol ** (1 / 3))
            self.out_size = (side, side, side)
            self.mode = "trilinear"

        self.out_dim = in_channels * math.prod(self.out_size)

        # Pad up the remaining gap to strictly match d_model
        self.pad = PadTensor((0, max(0, d_model - self.out_dim)))

    def forward(self, x):
        N, S, D = x.shape

        x_spatial = x.view(N * S, self.in_channels, *self.patch_size)

        x_down = F.interpolate(x_spatial, size=self.out_size, mode=self.mode)

        x_flat = x_down.view(N, S, self.out_dim)
        return self.pad(x_flat)
