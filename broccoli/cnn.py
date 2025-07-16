import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from einops import rearrange, repeat
import math
from typing import Type, Union, Tuple, Optional, Literal

from einops.layers.torch import Rearrange


# Helper function to calculate padding for 'same' mode
# Adapted from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
def _calculate_same_padding(
    input_size: Tuple[int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Calculates padding for 'same' output shape."""
    ih, iw = input_size
    kh, kw = kernel_size
    sh, sw = stride
    dh, dw = dilation

    # Effective kernel size
    eff_kh = (kh - 1) * dh + 1
    eff_kw = (kw - 1) * dw + 1

    # Calculate required total padding
    out_h = (ih + sh - 1) // sh
    out_w = (iw + sw - 1) // sw
    pad_h = max((out_h - 1) * sh + eff_kh - ih, 0)
    pad_w = max((out_w - 1) * sw + eff_kw - iw, 0)

    # Distribute padding (similar to TensorFlow 'SAME' behavior)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return (pad_left, pad_right, pad_top, pad_bottom)


# Custom Convolution Layer
class ConvLayer(nn.Module):
    """
    A 2D Convolution layer implemented using torch.nn.Unfold and a custom linear layer.

    This layer mimics the behavior of torch.nn.Conv2d but allows injecting
    a different linear layer implementation for processing the unfolded patches.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int, tuple or str, optional): Padding added to all four sides
            of the input. Can be an int, a tuple of two ints (padH, padW),
            a tuple of four ints (padLeft, padRight, padTop, padBottom),
            or the strings 'valid' (no padding) or 'same' (padding for same
            output spatial dims as input). Default: 0 ('valid').
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output.
            The bias is handled by the underlying linear layer. Default: True.
        linear (Type[nn.Module], optional): The class of the linear layer
            to use for the kernel operation. Must accept (in_features, out_features, bias)
            in its constructor. Defaults to torch.nn.Linear.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[
            int, Tuple[int, int], Tuple[int, int, int, int], Literal["valid", "same"]
        ] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        linear_module: Type[nn.Module] = nn.Linear,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.bias = bias
        self.linear_module = linear_module
        self.padding_mode = (
            padding  # Store the original padding mode ('same', 'valid', int, or tuple)
        )

        # Calculate the number of input features for the linear layer
        # It's the number of channels times the kernel area
        self.linear_in_features = (
            in_channels * self.kernel_size[0] * self.kernel_size[1]
        )

        # Instantiate the linear layer (kernel)
        self.kernel = self.linear_module(
            self.linear_in_features, out_channels, bias=bias
        )

        # We will use F.pad for manual padding, so unfold padding is 0
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,  # Manual padding handled in forward
            stride=self.stride,
        )

        # Determine numeric padding values for F.pad
        if isinstance(padding, str):
            if padding not in ["valid", "same"]:
                raise ValueError("padding must be 'valid', 'same', an int, or a tuple")
            # 'same' padding calculation depends on input size, defer to forward pass
            # 'valid' padding means 0
            self._padding_val = (
                (0, 0, 0, 0) if padding == "valid" else None
            )  # None indicates 'same'
        elif isinstance(padding, int):
            self._padding_val = (padding,) * 4
        elif isinstance(padding, tuple) and len(padding) == 2:
            # (padH, padW) -> (padW_left, padW_right, padH_top, padH_bottom)
            self._padding_val = (padding[1], padding[1], padding[0], padding[0])
        elif isinstance(padding, tuple) and len(padding) == 4:
            # (padLeft, padRight, padTop, padBottom) - already in F.pad format
            self._padding_val = padding
        else:
            raise TypeError(
                "padding must be 'valid', 'same', an int, or a tuple of 2 or 4 ints"
            )

    def _calculate_output_shape(self, h_in: int, w_in: int) -> Tuple[int, int]:
        """Calculates the output height and width."""
        if self._padding_val is None:  # 'same' padding
            # For 'same' padding, output size matches input size if stride is 1.
            # If stride > 1, output size is ceil(input_size / stride)
            # The _calculate_same_padding helper ensures this behavior.
            oh = math.ceil(h_in / self.stride[0])
            ow = math.ceil(w_in / self.stride[1])
            return oh, ow
        else:
            # Use the standard formula with the calculated numeric padding
            pad_h = self._padding_val[2] + self._padding_val[3]  # top + bottom
            pad_w = self._padding_val[0] + self._padding_val[1]  # left + right
            kh, kw = self.kernel_size
            sh, sw = self.stride
            dh, dw = self.dilation

            eff_kh = (kh - 1) * dh + 1
            eff_kw = (kw - 1) * dw + 1

            oh = math.floor((h_in + pad_h - eff_kh) / sh + 1)
            ow = math.floor((w_in + pad_w - eff_kw) / sw + 1)
            return oh, ow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H_in, W_in).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
        """
        N, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"Input channels {C} does not match expected {self.in_channels}"
            )

        # 1. Calculate and Apply Padding
        if self._padding_val is None:  # 'same' padding mode
            pad_l, pad_r, pad_t, pad_b = _calculate_same_padding(
                (H, W), self.kernel_size, self.stride, self.dilation
            )
            padded_x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            # Update H, W for output shape calculation after padding
            # Note: _calculate_output_shape will correctly handle 'same' based on original H, W
        elif self._padding_val != (0, 0, 0, 0):
            padded_x = F.pad(x, self._padding_val)
        else:  # No padding ('valid' or explicit 0)
            padded_x = x

        # 2. Unfold to extract patches
        # Input: (N, C_in, H_pad, W_pad)
        # Output: (N, C_in * K_h * K_w, L), where L is the number of patches (H_out * W_out)
        patches = self.unfold(padded_x)
        num_patches = patches.shape[-1]  # L

        # 3. Reshape for the linear layer
        # We want (N, L, C_in * K_h * K_w) to apply the linear layer patch-wise
        # transpose switches the last two dimensions
        patches_transposed = patches.transpose(1, 2)  # Shape: (N, L, C_in * K_h * K_w)

        # 4. Apply the linear layer (kernel) to each patch
        # Input: (N, L, linear_in_features)
        # Output: (N, L, out_channels)
        linear_output = self.kernel(patches_transposed)

        # 5. Reshape back to image format
        # We need (N, out_channels, L) first
        output_transposed = linear_output.transpose(1, 2)  # Shape: (N, out_channels, L)

        # Calculate output spatial dimensions
        out_h, out_w = self._calculate_output_shape(H, W)  # Use original H, W

        # Check if the number of patches matches the calculated output dimensions
        if num_patches != out_h * out_w:
            # This might happen with certain combinations of stride/padding/dilation/input size
            # if the calculation logic has an issue. nn.Unfold is usually robust.
            print(
                f"Warning: Mismatch in calculated patches. "
                f"Expected L={out_h * out_w}, got {num_patches}. "
                f"Using unfolded L={num_patches} to determine output shape."
            )
            # Attempt recovery if possible, though might indicate upstream calculation error
            # Find factors of num_patches close to expected out_h, out_w
            # This part is tricky and might not always yield the desired shape.
            # For simplicity, we'll rely on nn.Unfold's L and reshape.
            # A more robust solution might re-calculate H_out, W_out based *only* on L.
            # For now, let's stick to the reshape based on calculated out_h, out_w,
            # assuming they match L. If they don't, the reshape will fail.
            pass  # Proceed with calculated out_h, out_w

        # Reshape using einops (or tensor.view)
        # Input: (N, C_out, L) -> Output: (N, C_out, H_out, W_out)
        output = rearrange(output_transposed, "n c (h w) -> n c h w", h=out_h, w=out_w)
        # Alternative using view:
        # output = output_transposed.view(N, self.out_channels, out_h, out_w)

        return output

    def extra_repr(self) -> str:
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding_mode != 0 and self.padding_mode != "valid":
            s += ", padding={padding_mode}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        # if self.groups != 1: # Not implemented
        #     s += ', groups={groups}'
        if self.bias is False:
            s += ", bias=False"
        if self.linear_module != nn.Linear:
            s += f", linear={self.linear.__name__}"
        return s.format(**self.__dict__)


class WhiteningConv(ConvLayer):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        eigenvectors: torch.Tensor,
        bias: bool = True,
        linear_module: Type[nn.Module] = nn.Linear,
    ):
        """
        We end up using a concatenation of the eigenvector tensor with its negation,
            as the tendency to use e.g. ReLU in neural networks means that useful
            data may otherwise be lost (if one orientation of an eigenvector produces
            a strong negative signal, this will be clipped to zero by ReLU, but a
            strong positive signal from the negation of the eigenvector will be
            preserved). Assuming a square kernel, out channels is thus

            (kernel_size ** 2) * in_channels * 2

            where the trailing "* 2" accounts for the doubling of the size of the
            eigenvector tensor we're using by including the negative of each eigenvector
            as well.
        """
        out_channels = kernel_size**2 * in_channels * 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=bias,
            linear_module=linear_module,
        )
        self.eigenvectors = torch.cat([eigenvectors, -eigenvectors], dim=0)
        # bias updates if `bias`=True but weight doesn't,
        #   per Jordan (2024) https://arxiv.org/abs/2404.00498
        #   but weight is set to `requires_grad = False`:
        # self.kernel.weight.requires_grad = False
        with torch.no_grad():
            self.kernel.weight.copy_(self.eigenvectors)
        assert self.kernel.weight.requires_grad


class ConcatPool(nn.Module):
    """
    A layer that concatenates nearby squares of an image
    """

    def __init__(
        self,
        in_channels: int,
        pooling_kernel_size: int,
        pooling_kernel_stride: int,
        pooling_kernel_padding: int,
        d_model: int,
        activation_kwargs: Optional[dict] = None,
        activation: nn.Module = nn.ReLU,
        linear_module: Type[nn.Module] = nn.Linear,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_kernel_stride = pooling_kernel_stride
        self.pooling_kernel_padding = pooling_kernel_padding
        self.d_model = d_model
        self.pooling_output_size = (pooling_kernel_size**2) * self.in_channels
        if activation_kwargs is not None:
            self.activation = activation(**activation_kwargs)
        else:
            self.activation = activation()
        self.process = nn.Sequential(
            *[
                nn.Unfold(
                    pooling_kernel_size,
                    stride=pooling_kernel_stride,
                    padding=pooling_kernel_padding,
                ),
                Rearrange("N Block L -> N L Block"),
                linear_module(
                    self.pooling_output_size,
                    2 * d_model if activation.__name__.endswith("GLU") else d_model,
                ),
                self.activation,
            ]
        )

    def forward(self, x):
        return self.process(x)


class WhitenConcatProject(nn.Module):
    """
    A layer that whitens patches of the input image, then concatenates nearby
        patch embeddings before rearranging the outputs into the format expected
        by a Transformer block
    """

    def __init__(
        self,
        in_channels: int,
        convolution_kernel_size: int,
        pooling_kernel_size: int,
        pooling_kernel_stride: int,
        pooling_kernel_padding: int,
        d_model: int,
        dropout=0.0,
        activation: nn.Module = nn.ReLU,
        activation_kwargs: Optional[dict] = None,
        eigenvectors: torch.Tensor = None,
        bias: bool = True,
        linear_module: Type[nn.Module] = nn.Linear,
    ):
        super().__init__()
        assert eigenvectors is not None
        self.in_channels = in_channels
        self.convolution_kernel_size = convolution_kernel_size
        self.pooling_kernel_size = pooling_kernel_size
        self.whitening_out_channels = convolution_kernel_size**2 * in_channels * 2
        self.pooling_output_size = (
            pooling_kernel_size**2
        ) * self.whitening_out_channels
        self.process = nn.Sequential(
            *[
                WhiteningConv(
                    in_channels,
                    convolution_kernel_size,
                    eigenvectors,
                    bias=bias,
                    linear_module=linear_module,
                ),
                ConcatPool(
                    self.whitening_out_channels,
                    pooling_kernel_size,
                    pooling_kernel_stride,
                    pooling_kernel_padding,
                    d_model,
                    activation,
                    activation_kwargs,
                    linear_module,
                ),
                nn.Dropout(dropout),
            ]
        )

    def forward(self, x):
        return self.process(x)
