import math
from typing import Optional

from .transformer import TransformerEncoder, DenoisingAutoEncoder
from .cnn import SpaceToDepth, calculate_output_spatial_size, spatial_tuple
from .activation import ReLU, SquaredReLU, GELU, SwiGLU
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
        return F.pad(x, *self.args, **self.kwargs)


class SequencePool(nn.Module):
    """
    As described in [Hasani et al. (2021) *''Escaping the Big Data Paradigm with
        Compact Transformers''*](https://arxiv.org/abs/2104.05704). It can be viewed
        as a generalisation of average pooling.
    """

    def __init__(self, d_model, linear_module, out_dim, batch_norm=True):
        super().__init__()
        self.d_model = d_model
        self.attention = nn.Sequential(
            *[
                linear_module(d_model, 1),
                Rearrange("batch seq 1 -> batch seq"),
                nn.Softmax(dim=-1),
            ]
        )
        self.projection = nn.Linear(d_model, out_dim)
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm = nn.BatchNorm1d(out_dim, affine=False)
        else:
            self.norm = None

    def forward(self, x):
        weights = self.attention(x)
        weighted_embedding = einsum(
            weights, x, "batch seq, batch seq d_model -> batch d_model"
        )
        projection = self.projection(weighted_embedding)
        return self.norm(projection) if self.batch_norm else projection


class DCTEncoder(nn.Module):
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
        cnn_in_channels=3,
        minimum_cnn_out_channels=16,
        cnn_kernel_size=3,
        cnn_kernel_stride=1,
        cnn_padding="same",
        cnn_kernel_dilation=1,
        cnn_kernel_groups=1,
        cnn_activation: nn.Module = ReLU,
        cnn_activation_kwargs: Optional[dict] = None,
        cnn_dropout=0.0,
        pooling_type="concat",  # maxpool or concat
        pooling_kernel_size=3,
        pooling_kernel_stride=2,
        pooling_padding=1,
        transformer_position_embedding="relative",  # absolute or relative
        transformer_embedding_size=256,
        transformer_layers=7,
        transformer_heads=4,
        transformer_mlp_ratio=2,
        transformer_bos_tokens=0,
        transformer_activation: nn.Module = SquaredReLU,
        transformer_activation_kwargs: Optional[dict] = None,
        mlp_dropout=0.0,
        msa_dropout=0.1,
        stochastic_depth=0.1,
        linear_module=nn.Linear,
        initial_batch_norm=True,
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
            convxd = nn.Conv1d
            batchnormxd = nn.BatchNorm1d
            spatial_dim_names = "D1"
        elif self.spatial_dimensions == 2:
            maxpoolxd = nn.MaxPool2d
            convxd = nn.Conv2d
            batchnormxd = nn.BatchNorm2d
            spatial_dim_names = "D1 D2"
        elif self.spatial_dimensions == 3:
            maxpoolxd = nn.MaxPool3d
            convxd = nn.Conv3d
            batchnormxd = nn.BatchNorm3d
            spatial_dim_names = "D1 D2 D3"
        else:
            raise NotImplementedError(
                "`input_size` must be a tuple of length 1, 2, or 3."
            )

        cnn_output_size = calculate_output_spatial_size(
            input_size,
            kernel_size=cnn_kernel_size,
            stride=cnn_kernel_stride,
            padding=cnn_padding,
            dilation=cnn_kernel_dilation,
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

        self.sequence_length = math.prod(pooling_output_size)  # One token per voxel

        pooling_kernel_voxels = math.prod(
            spatial_tuple(pooling_kernel_size, self.spatial_dimensions)
        )

        if pooling_type in ["maxpool", None]:
            cnn_out_channels = transformer_embedding_size
        elif pooling_type == "concat":
            cnn_out_channels = min(
                math.floor(transformer_embedding_size / pooling_kernel_voxels),
                minimum_cnn_out_channels,
            )
        else:
            raise NotImplementedError("Pooling type must be maxpool, concat or None")

        cnn_activation_out_channels = cnn_out_channels

        # This block rhymes:
        if cnn_activation.__name__.endswith("GLU"):
            cnn_out_channels *= 2

        self.cnn = convxd(
            cnn_in_channels,
            cnn_out_channels,
            cnn_kernel_size,
            stride=cnn_kernel_stride,
            padding=cnn_padding,
            dilation=cnn_kernel_dilation,
            groups=cnn_kernel_groups,
            bias=True,
            padding_mode="zeros",
        )

        self.activate_and_dropout = nn.Sequential(
            *[
                Rearrange(  # rearrange in case we're using XGLU activation
                    f"N C {spatial_dim_names} -> N {spatial_dim_names} C"
                ),
                self.cnn_activation,
                Rearrange(f"N {spatial_dim_names} C -> N C {spatial_dim_names}"),
                nn.Dropout(cnn_dropout),
                (
                    batchnormxd(cnn_activation_out_channels)
                    if initial_batch_norm
                    else nn.Identity()
                ),
            ]
        )

        if pooling_type is None:
            self.pool = nn.Sequential(
                *[
                    Rearrange(
                        f"N C {spatial_dim_names} -> N ({spatial_dim_names}) C"
                    ),  # for transformer
                ]
            )

        elif pooling_type == "maxpool":
            self.pool = nn.Sequential(
                *[
                    maxpoolxd(
                        pooling_kernel_size,
                        stride=pooling_kernel_stride,
                        padding=pooling_padding,
                    ),
                    Rearrange(
                        f"N C {spatial_dim_names} -> N ({spatial_dim_names}) C"
                    ),  # for transformer
                ]
            )

        elif pooling_type == "concat":

            if transformer_activation_kwargs is not None:
                self.concatpool_activation = transformer_activation(
                    **transformer_activation_kwargs
                )
            else:
                self.concatpool_activation = transformer_activation()

            concatpool_out_channels = (
                pooling_kernel_voxels * cnn_activation_out_channels
            )

            self.pool = nn.Sequential(
                *[
                    SpaceToDepth(
                        pooling_kernel_size,
                        stride=pooling_kernel_stride,
                        padding=pooling_padding,
                        spatial_dimensions=self.spatial_dimensions,
                    ),
                    Rearrange(  # for transformer
                        f"N C {spatial_dim_names} -> N ({spatial_dim_names}) C"
                    ),
                    DenoisingAutoEncoder(
                        concatpool_out_channels,
                        transformer_mlp_ratio,
                        transformer_embedding_size,
                        activation=transformer_activation,
                        activation_kwargs=transformer_activation_kwargs,
                        dropout=0.0,
                        linear_module=linear_module,
                    ),
                ]
            )

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
                mlp_dropout=mlp_dropout,
                msa_dropout=msa_dropout,
                stochastic_depth=stochastic_depth,
                causal=False,
                linear_module=linear_module,
                bos_tokens=transformer_bos_tokens,
            )
        else:
            self.transformer = nn.Identity()

        self.encoder = nn.Sequential(
            *[
                batchnormxd(cnn_in_channels) if initial_batch_norm else nn.Identity(),
                self.cnn,
                self.activate_and_dropout,
                self.pool,
                self.transformer,
            ]
        )

    def forward(self, x):
        return self.encoder(x)


class DCT(nn.Module):
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
        cnn_in_channels=3,
        minimum_cnn_out_channels=16,
        cnn_kernel_size=3,
        cnn_kernel_stride=1,
        cnn_padding="same",
        cnn_kernel_dilation=1,
        cnn_kernel_groups=1,
        cnn_activation: nn.Module = ReLU,
        cnn_activation_kwargs: Optional[dict] = None,
        cnn_dropout=0.0,
        pooling_type="concat",  # maxpool or concat
        pooling_kernel_size=3,
        pooling_kernel_stride=2,
        pooling_padding=1,
        transformer_position_embedding="relative",  # absolute or relative
        transformer_embedding_size=256,
        transformer_layers=7,
        transformer_heads=4,
        transformer_mlp_ratio=2,
        transformer_bos_tokens=0,
        transformer_activation: nn.Module = SquaredReLU,
        transformer_activation_kwargs: Optional[dict] = None,
        mlp_dropout=0.0,
        msa_dropout=0.1,
        stochastic_depth=0.1,
        batch_norm_outputs=True,
        linear_module=nn.Linear,
        initial_batch_norm=True,
        image_classes=100,
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

        self.encoder = DCTEncoder(
            input_size=input_size,
            cnn_in_channels=cnn_in_channels,
            minimum_cnn_out_channels=minimum_cnn_out_channels,
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
            transformer_position_embedding=transformer_position_embedding,
            transformer_embedding_size=transformer_embedding_size,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_mlp_ratio=transformer_mlp_ratio,
            transformer_bos_tokens=transformer_bos_tokens,
            transformer_activation=transformer_activation,
            transformer_activation_kwargs=transformer_activation_kwargs,
            mlp_dropout=mlp_dropout,
            msa_dropout=msa_dropout,
            stochastic_depth=stochastic_depth,
            linear_module=linear_module,
            initial_batch_norm=initial_batch_norm,
        )
        self.pool = SequencePool(
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
