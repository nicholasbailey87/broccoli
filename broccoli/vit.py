import math
from typing import Optional

from .transformer import TransformerEncoder, FeedforwardBlock
from .cnn import SpaceToDepth, calculate_output_spatial_size, spatial_tuple
from .activation import ReLU, SquaredReLU, GELU, SwiGLU
from .linear import SpectralNormLinear
from einops import einsum
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch.nn.functional as F


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


class GetCLSToken(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, 0, :]


class SequencePool(nn.Module):
    def __init__(self, d_model, linear_module=nn.Linear):
        super().__init__()
        self.attention = nn.Sequential(
            *[
                linear_module(d_model, 1),
                Rearrange("batch seq 1 -> batch seq"),
                nn.Softmax(dim=-1),
            ]
        )

    def forward(self, x):
        weights = self.attention(x)
        return einsum(weights, x, "batch seq, batch seq d_model -> batch d_model")


class ClassificationHead(nn.Module):
    """
    A general classification head for a ViT
    """

    def __init__(self, d_model, linear_module, n_classes, batch_norm=True):
        super().__init__()
        self.d_model = d_model
        self.summarize = GetCLSToken()
        self.projection = linear_module(d_model, n_classes)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(n_classes, affine=False)
        else:
            self.batch_norm = nn.Identity()

        self.classification_process = nn.Sequential(
            *[
                self.summarize,
                self.projection,
                self.batch_norm,
            ]
        )

    def forward(self, x):
        return self.classification_process(x)


class SequencePoolClassificationHead(ClassificationHead):
    """
    As described in [Hasani et al. (2021) *''Escaping the Big Data Paradigm with
        Compact Transformers''*](https://arxiv.org/abs/2104.05704). It can be viewed
        as a generalisation of average pooling.
    """

    def __init__(self, d_model, linear_module, out_dim, batch_norm=True):
        super().__init__(d_model, linear_module, out_dim, batch_norm=batch_norm)
        self.summarize = SequencePool(d_model, linear_module)


class ViTEncoder(nn.Module):
    """
    Based on the Compact Convolutional Transformer (CCT) of [Hasani et al. (2021)
        *''Escaping the Big Data Paradigm with Compact Transformers''*](
        https://arxiv.org/abs/2104.05704). It's basically a convolutional neural
        network leading into a transformer encoder. To make it like the full CCT
        we would finish it of with a sequence pooling layer but we won't always
        want to do that.
    """

    def __init__(
        self,
        input_size=(32, 32),
        in_channels=3,
        initial_batch_norm=True,
        cnn=True,
        cnn_out_channels=16,
        cnn_kernel_size=3,
        cnn_kernel_stride=1,
        cnn_padding="same",
        cnn_kernel_dilation=1,
        cnn_kernel_groups=1,
        cnn_activation: nn.Module = ReLU,
        cnn_activation_kwargs: Optional[dict] = None,
        cnn_dropout=0.0,
        pooling_type="concat",  # max, average or concat
        pooling_kernel_size=3,
        pooling_kernel_stride=2,
        pooling_padding=1,
        transformer_feedforward_first=True,
        transformer_initial_ff_residual_path=True,
        transformer_initial_ff_linear_module_up=None,
        transformer_initial_ff_linear_module_down=None,
        transformer_pre_norm=True,
        transformer_normformer=False,
        transformer_post_norm=False,
        transformer_position_embedding="relative",  # absolute or relative
        transformer_embedding_size=256,
        transformer_layers=7,
        transformer_heads=4,
        transformer_mlp_ratio=2,
        transformer_bos_tokens=0,
        transformer_return_bos_tokens=False,
        transformer_activation: nn.Module = SquaredReLU,
        transformer_activation_kwargs: Optional[dict] = None,
        transformer_mlp_dropout=0.0,
        transformer_msa_dropout=0.1,
        transformer_stochastic_depth=0.1,
        linear_module=nn.Linear,
    ):
        super().__init__()

        if cnn_activation_kwargs is not None:
            self.cnn_activation = cnn_activation(**cnn_activation_kwargs)
        else:
            self.cnn_activation = cnn_activation()

        if transformer_activation_kwargs is not None:
            self.transformer_activation = transformer_activation(
                **transformer_activation_kwargs
            )
        else:
            self.transformer_activation = transformer_activation()

        self.input_size = input_size
        self.spatial_dimensions = len(self.input_size)

        if self.spatial_dimensions == 1:
            maxpoolxd = nn.MaxPool1d
            avgpoolxd = nn.AvgPool1d
            convxd = nn.Conv1d
            batchnormxd = nn.BatchNorm1d
            spatial_dim_names = "D1"
        elif self.spatial_dimensions == 2:
            maxpoolxd = nn.MaxPool2d
            avgpoolxd = nn.AvgPool2d
            convxd = nn.Conv2d
            batchnormxd = nn.BatchNorm2d
            spatial_dim_names = "D1 D2"
        elif self.spatial_dimensions == 3:
            maxpoolxd = nn.MaxPool3d
            avgpoolxd = nn.AvgPool3d
            convxd = nn.Conv3d
            batchnormxd = nn.BatchNorm3d
            spatial_dim_names = "D1 D2 D3"
        else:
            raise NotImplementedError(
                "`input_size` must be a tuple of length 1, 2, or 3."
            )

        if cnn:
            # This block rhymes:
            if cnn_activation.__name__.endswith("GLU"):
                cnn_out_channels *= 2
            cnn_output_size = calculate_output_spatial_size(
                input_size,
                kernel_size=cnn_kernel_size,
                stride=cnn_kernel_stride,
                padding=cnn_padding,
                dilation=cnn_kernel_dilation,
            )
            self.cnn = convxd(
                in_channels,
                cnn_out_channels,
                cnn_kernel_size,
                stride=cnn_kernel_stride,
                padding=cnn_padding,
                dilation=cnn_kernel_dilation,
                groups=cnn_kernel_groups,
                bias=True,
                padding_mode="zeros",
            )
            cnn_activation_out_channels = cnn_out_channels
            self.activate_and_dropout = nn.Sequential(
                *[
                    Rearrange(  # rearrange in case we're using XGLU activation
                        f"N C {spatial_dim_names} -> N {spatial_dim_names} C"
                    ),
                    self.cnn_activation,
                    Rearrange(f"N {spatial_dim_names} C -> N C {spatial_dim_names}"),
                    nn.Dropout(cnn_dropout),
                    batchnormxd(cnn_activation_out_channels),
                ]
            )
        else:
            self.cnn = nn.Identity()
            self.activate_and_dropout = nn.Identity()
            cnn_output_size = input_size
            cnn_out_channels = in_channels
            cnn_activation_out_channels = in_channels

        pooling_kernel_voxels = math.prod(
            spatial_tuple(pooling_kernel_size, self.spatial_dimensions)
        )

        pooling_output_size = (
            cnn_output_size
            if pooling_type is None
            else calculate_output_spatial_size(
                cnn_output_size,
                kernel_size=pooling_kernel_size,
                stride=pooling_kernel_stride,
                padding=pooling_padding,
                dilation=1,
            )
        )

        if pooling_type is None:
            pooling_out_channels = cnn_activation_out_channels
            self.pool = nn.Sequential(
                *[
                    Rearrange(
                        f"N C {spatial_dim_names} -> N ({spatial_dim_names}) C"
                    ),  # for transformer
                ]
            )

        elif pooling_type == "max":
            pooling_out_channels = cnn_activation_out_channels
            self.pool = maxpoolxd(
                pooling_kernel_size,
                stride=pooling_kernel_stride,
                padding=pooling_padding,
            )
        elif pooling_type == "average":
            pooling_out_channels = cnn_activation_out_channels
            self.pool = avgpoolxd(
                pooling_kernel_size,
                stride=pooling_kernel_stride,
                padding=pooling_padding,
            )
        elif pooling_type == "concat":
            pooling_out_channels = pooling_kernel_voxels * cnn_activation_out_channels
            self.pool = SpaceToDepth(
                pooling_kernel_size,
                stride=pooling_kernel_stride,
                padding=pooling_padding,
                spatial_dimensions=self.spatial_dimensions,
            )
        else:
            raise NotImplementedError(
                "Pooling type must be max, average, concat or None"
            )

        self.pooling_channels_padding = PadTensor(
            (0, max(0, transformer_embedding_size - pooling_out_channels))
        )

        self.sequence_length = math.prod(pooling_output_size)  # One token per voxel

        if transformer_layers > 0:
            self.transformer = TransformerEncoder(
                self.sequence_length,
                transformer_embedding_size,
                transformer_layers,
                transformer_heads,
                position_embedding_type=transformer_position_embedding,
                source_size=pooling_output_size,
                mlp_ratio=transformer_mlp_ratio,
                activation=transformer_activation,
                activation_kwargs=transformer_activation_kwargs,
                mlp_dropout=transformer_mlp_dropout,
                msa_dropout=transformer_msa_dropout,
                stochastic_depth=transformer_stochastic_depth,
                causal=False,
                linear_module=linear_module,
                bos_tokens=transformer_bos_tokens,
                return_bos_tokens=transformer_return_bos_tokens,
                pre_norm=transformer_pre_norm,
                normformer=transformer_normformer,
                post_norm=transformer_post_norm,
            )
        else:
            self.transformer = nn.Identity()

        if transformer_feedforward_first:
            self.initial_ff = FeedforwardBlock(
                max(transformer_embedding_size, pooling_out_channels),
                transformer_mlp_ratio,
                transformer_embedding_size,
                activation=transformer_activation,
                activation_kwargs=transformer_activation_kwargs,
                dropout=transformer_mlp_dropout,
                linear_module_up=(
                    transformer_initial_ff_linear_module_up
                    if transformer_initial_ff_linear_module_up is not None
                    else linear_module
                ),
                linear_module_down=(
                    transformer_initial_ff_linear_module_down
                    if transformer_initial_ff_linear_module_down is not None
                    else linear_module
                ),
                pre_norm=transformer_pre_norm,
                normformer=transformer_normformer,
                post_norm=transformer_post_norm,
                residual_path=transformer_initial_ff_residual_path,
            )
        else:
            self.initial_ff = nn.Identity()

        self.encoder = nn.Sequential(
            *[
                batchnormxd(in_channels) if initial_batch_norm else nn.Identity(),
                self.cnn,
                self.activate_and_dropout,
                self.pool,
                Rearrange(  # for transformer
                    f"N C {spatial_dim_names} -> N ({spatial_dim_names}) C"
                ),
                self.pooling_channels_padding,
                self.initial_ff,
                self.transformer,
            ]
        )

    def forward(self, x):
        return self.encoder(x)


class ViT(nn.Module):
    """
    Denoising convolutional transformer
    Based on the Compact Convolutional Transformer (CCT) of [Hasani et al. (2021)
        *''Escaping the Big Data Paradigm with Compact Transformers''*](
        https://arxiv.org/abs/2104.05704). It's a convolutional neural network
        leading into a transformer encoder, followed by a sequence pooling layer.
    """

    def __init__(
        self,
        input_size=(32, 32),
        image_classes=100,
        in_channels=3,
        initial_batch_norm=True,
        cnn=True,
        cnn_out_channels=16,
        cnn_kernel_size=3,
        cnn_kernel_stride=1,
        cnn_padding="same",
        cnn_kernel_dilation=1,
        cnn_kernel_groups=1,
        cnn_activation: nn.Module = ReLU,
        cnn_activation_kwargs: Optional[dict] = None,
        cnn_dropout=0.0,
        pooling_type="concat",  # max, average or concat
        pooling_kernel_size=3,
        pooling_kernel_stride=2,
        pooling_padding=1,
        transformer_feedforward_first=True,
        transformer_initial_ff_residual_path=True,
        transformer_initial_ff_linear_module_up=None,
        transformer_initial_ff_linear_module_down=None,
        transformer_pre_norm=True,
        transformer_normformer=False,
        transformer_post_norm=False,
        transformer_position_embedding="relative",  # absolute or relative
        transformer_embedding_size=256,
        transformer_layers=7,
        transformer_heads=4,
        transformer_mlp_ratio=2,
        transformer_bos_tokens=0,
        transformer_return_bos_tokens=False,
        transformer_activation: nn.Module = SquaredReLU,
        transformer_activation_kwargs: Optional[dict] = None,
        transformer_mlp_dropout=0.0,
        transformer_msa_dropout=0.1,
        transformer_stochastic_depth=0.1,
        batch_norm_outputs=True,
        linear_module=SpectralNormLinear,
        head=SequencePoolClassificationHead,
    ):

        super().__init__()

        if isinstance(cnn_activation, str):
            cnn_activation = {
                "ReLU": ReLU,
                "SquaredReLU": SquaredReLU,
                "GELU": GELU,
                "SwiGLU": SwiGLU,
            }[cnn_activation]

        if isinstance(transformer_activation, str):
            transformer_activation = {
                "ReLU": ReLU,
                "SquaredReLU": SquaredReLU,
                "GELU": GELU,
                "SwiGLU": SwiGLU,
            }[transformer_activation]

        self.encoder = ViTEncoder(
            input_size=input_size,
            initial_batch_norm=initial_batch_norm,
            in_channels=in_channels,
            cnn=cnn,
            cnn_out_channels=cnn_out_channels,
            cnn_kernel_size=cnn_kernel_size,
            cnn_kernel_stride=cnn_kernel_stride,
            cnn_padding=cnn_padding,
            cnn_kernel_dilation=cnn_kernel_dilation,
            cnn_kernel_groups=cnn_kernel_groups,
            cnn_activation=cnn_activation,
            cnn_activation_kwargs=cnn_activation_kwargs,
            cnn_dropout=cnn_dropout,
            pooling_type=pooling_type,
            pooling_kernel_size=pooling_kernel_size,
            pooling_kernel_stride=pooling_kernel_stride,
            pooling_padding=pooling_padding,
            transformer_feedforward_first=transformer_feedforward_first,
            transformer_initial_ff_residual_path=transformer_initial_ff_residual_path,
            transformer_initial_ff_linear_module_up=transformer_initial_ff_linear_module_up,
            transformer_initial_ff_linear_module_down=transformer_initial_ff_linear_module_down,
            transformer_pre_norm=transformer_pre_norm,
            transformer_normformer=transformer_normformer,
            transformer_post_norm=transformer_post_norm,
            transformer_position_embedding=transformer_position_embedding,
            transformer_embedding_size=transformer_embedding_size,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_mlp_ratio=transformer_mlp_ratio,
            transformer_bos_tokens=transformer_bos_tokens,
            transformer_return_bos_tokens=transformer_return_bos_tokens,
            transformer_activation=transformer_activation,
            transformer_activation_kwargs=transformer_activation_kwargs,
            transformer_mlp_dropout=transformer_mlp_dropout,
            transformer_msa_dropout=transformer_msa_dropout,
            transformer_stochastic_depth=transformer_stochastic_depth,
            linear_module=linear_module,
        )

        self.pool = head(
            transformer_embedding_size,
            linear_module,
            image_classes,
            batch_norm=batch_norm_outputs,
        )

    @property
    def sequence_length(self):
        return self.encoder.sequence_length

    def forward(self, x):
        return self.pool(self.encoder(x))
