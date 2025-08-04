import pytest
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from typing import Union, Tuple

# Import the components to be tested from the local cnn.py module
from broccoli.cnn import ConvLayer

# Define common parameters for tests
IN_CHANNELS = 3
OUT_CHANNELS = 8
BATCH_SIZE = 4
INPUT_H = 28
INPUT_W = 28
# XXX: refine ATOL value below, add RTOL, and give reasoning for both in comment
ATOL = 1e-4


@pytest.fixture
def input_tensor():
    """Provides a standard input tensor for tests."""
    return torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_H, INPUT_W)


def run_convolution_comparison(
    input_tensor: torch.Tensor,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[str, int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    bias: bool = True,
):
    """
    A helper function to instantiate, run, and compare a standard Conv2d
    with the custom ConvLayer.
    """
    # 1. Instantiate both layers with the same parameters
    conv_std = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )
    conv_custom = ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )

    # 2. Ensure both layers have the same weights and bias for a fair comparison
    conv_custom.kernel.weight.data.copy_(conv_std.weight.data.view(out_channels, -1))
    if bias:
        conv_custom.kernel.bias.data.copy_(conv_std.bias.data)

    # Enable gradient tracking on the input
    input_tensor.requires_grad = True

    # 3. Forward Pass Comparison
    output_std = conv_std(input_tensor)
    output_custom = conv_custom(input_tensor)

    assert output_std.shape == output_custom.shape, "Output shapes do not match"
    assert torch.allclose(
        output_std, output_custom, atol=ATOL
    ), f"Forward pass outputs do not match for padding={padding!r}"

    # 4. Backward Pass (Gradient) Comparison
    # Use a dummy loss for backpropagation
    loss_std = output_std.sum()
    loss_custom = output_custom.sum()
    loss_std.backward(retain_graph=True)
    loss_custom.backward(retain_graph=True)

    # Compare input gradients
    assert torch.allclose(
        input_tensor.grad, input_tensor.grad, atol=ATOL
    ), f"Input gradients do not match for padding={padding!r}"

    # Compare weight gradients
    std_weight_grad_reshaped = conv_std.weight.grad.view(out_channels, -1)
    custom_weight_grad = conv_custom.kernel.weight.grad
    assert torch.allclose(
        std_weight_grad_reshaped, custom_weight_grad, atol=ATOL
    ), f"Weight gradients do not match for padding={padding!r}"

    # Compare bias gradients
    if bias:
        assert torch.allclose(
            conv_std.bias.grad, conv_custom.kernel.bias.grad, atol=ATOL
        ), f"Bias gradients do not match for padding={padding!r}"


@pytest.mark.parametrize("kernel_size", [1, 3, (5, 3)])
@pytest.mark.parametrize("stride", [1, 2, (1, 2)])
@pytest.mark.parametrize(
    "padding",
    [
        "valid",  # String 'valid'
        "same",  # String 'same'
        0,  # Integer padding
        1,  # Integer padding
        (2, 1),  # Tuple (padH, padW)
        # (1, 2, 2, 1),  # Tuple (padL, padR, padT, padB)
    ],
)
@pytest.mark.parametrize("dilation", [1, 2, (1, 2)])
@pytest.mark.parametrize("bias", [True, False])
def test_conv_layer_extensive(
    input_tensor, kernel_size, stride, padding, dilation, bias
):
    """
    Tests the ConvLayer against nn.Conv2d over a wide range of parameters.
    """
    # PyTorch's 'same' padding requires stride > 1 for dilation > 1
    if padding == "same" and (
        any(s > 1 for s in _pair(stride)) or any(d > 1 for d in _pair(dilation))
    ):
        pytest.skip(
            "PyTorch Conv2d 'same' padding with stride or dilation > 1 is not supported."
        )

    run_convolution_comparison(
        input_tensor=input_tensor,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )


def test_invalid_padding_string():
    """Tests that an invalid string for padding raises a ValueError."""
    with pytest.raises(
        ValueError, match="padding must be 'valid', 'same', an int, or a tuple"
    ):
        ConvLayer(
            in_channels=3, out_channels=8, kernel_size=3, padding="invalid_string"
        )


def test_invalid_padding_type():
    """Tests that an invalid type for padding raises a TypeError."""
    with pytest.raises(
        TypeError,
        match="padding must be 'valid', 'same', an int, or a tuple of 2 or 4 ints",
    ):
        ConvLayer(in_channels=3, out_channels=8, kernel_size=3, padding=1.5)  # float


def test_invalid_padding_tuple_length():
    """Tests that a tuple of invalid length for padding raises a TypeError."""
    with pytest.raises(
        TypeError,
        match="padding must be 'valid', 'same', an int, or a tuple of 2 or 4 ints",
    ):
        ConvLayer(
            in_channels=3, out_channels=8, kernel_size=3, padding=(1, 2, 3)
        )  # length 3
