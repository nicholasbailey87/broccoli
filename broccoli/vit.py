import math
from typing import Optional

from .transformer import TransformerEncoder
from .cnn import ConvLayer, ConcatPool
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

    def __init__(self, d_model, linear_module, out_dim):
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
        self.norm = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        weights = self.attention(x)
        weighted_embedding = einsum(
            weights, x, "batch seq, batch seq d_model -> batch d_model"
        )
        projection = self.projection(weighted_embedding)
        return self.norm(projection)


class CCTEncoder(nn.Module):
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
        image_size=32,
        conv_kernel_size=3,
        conv_pooling_type="maxpool",
        conv_pooling_kernel_size=3,
        conv_pooling_kernel_stride=2,
        conv_pooling_kernel_padding=1,
        conv_dropout=0.0,
        transformer_position_embedding="absolute",  # absolute or relative
        transformer_embedding_size=256,
        transformer_layers=7,
        transformer_heads=4,
        transformer_mlp_ratio=2,
        transformer_bos_tokens=4,
        tranformer_share_kv=True,
        tranformer_max_subtract=True,
        tranformer_d_model_scale=True,
        tranformer_log_length_scale=True,
        tranformer_quiet_attention=True,
        cnn_activation: nn.Module = nn.ReLU,
        cnn_activation_kwargs: Optional[dict] = None,
        transformer_activation: nn.Module = nn.GELU,
        transformer_activation_kwargs: Optional[dict] = None,
        mlp_dropout=0.0,
        msa_dropout=0.1,
        stochastic_depth=0.1,
        linear_module=nn.Linear,
        image_channels=3,
        batch_norm=False,
    ):
        if conv_pooling_type not in ["maxpool", "concat"]:
            raise NotImplementedError("Pooling type must be maxpool or concat")

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

        self.image_size = image_size

        # XXX: We assume a square image here
        output_size = math.floor(
            (image_size + 2 * conv_pooling_kernel_padding - conv_pooling_kernel_size)
            / conv_pooling_kernel_stride
            + 1
        )  # output of pooling

        self.sequence_length = output_size**2

        if conv_pooling_type == "maxpool":
            conv_out_channels = transformer_embedding_size
        elif conv_pooling_type == "concat":
            conv_out_channels = int(
                math.floor(transformer_embedding_size / (conv_pooling_kernel_size**2))
            )

        # This if block rhymes:
        if cnn_activation.__name__.endswith("GLU"):
            conv_out_channels *= 2

        self.conv = ConvLayer(
            image_channels,
            conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=1,
            padding="same",
            linear_module=linear_module,
        )

        if conv_pooling_type == "maxpool":
            self.pool = nn.Sequential(
                *[
                    Rearrange(  # rearrange in case we're using XGLU activation
                        "N C H W -> N H W C"
                    ),
                    self.cnn_activation,
                    Rearrange("N H W C -> N C H W"),
                    nn.MaxPool2d(
                        conv_pooling_kernel_size,
                        stride=conv_pooling_kernel_stride,
                        padding=conv_pooling_kernel_padding,
                    ),
                    Rearrange("N C H W -> N (H W) C"),  # for transformer
                ]
            )

        elif conv_pooling_type == "concat":

            if transformer_activation_kwargs is not None:
                self.concatpool_activation = transformer_activation(
                    **transformer_activation_kwargs
                )
            else:
                self.concatpool_activation = transformer_activation()

            concatpool_out_channels = conv_pooling_kernel_size**2 * conv_out_channels

            if cnn_activation.__name__.endswith("GLU"):
                cnn_activation_output_channels = concatpool_out_channels / 2
            else:
                cnn_activation_output_channels = concatpool_out_channels

            self.pool = nn.Sequential(
                *[
                    ConcatPool(
                        conv_pooling_kernel_size,
                        stride=conv_pooling_kernel_stride,
                        padding=conv_pooling_kernel_padding,
                    ),
                    Rearrange(  # rearrange in case we're using XGLU activation
                        "N C H W -> N H W C"
                    ),
                    self.cnn_activation,
                    nn.Dropout(conv_dropout),
                    Rearrange(  # rearrange in case we're using XGLU activation
                        "N H W C -> N C H W"
                    ),
                    nn.BatchNorm2d(cnn_activation_output_channels),
                    Rearrange(  # rearrange in case we're using XGLU activation
                        "N C H W -> N (H W) C"
                    ),
                    nn.Linear(
                        cnn_activation_output_channels,
                        (
                            2 * transformer_embedding_size * transformer_mlp_ratio
                            if transformer_activation.__name__.endswith("GLU")
                            else transformer_embedding_size * transformer_mlp_ratio
                        ),
                    ),
                    self.concatpool_activation,
                    nn.Linear(transformer_embedding_size * transformer_mlp_ratio),
                ]
            )

        if transformer_layers > 0:
            self.transformer = TransformerEncoder(
                self.sequence_length,
                transformer_embedding_size,
                transformer_layers,
                transformer_heads,
                position_embedding_type=transformer_position_embedding,
                source_size=(output_size, output_size),
                mlp_ratio=transformer_mlp_ratio,
                activation=transformer_activation,
                activation_kwargs=transformer_activation_kwargs,
                mlp_dropout=mlp_dropout,
                msa_dropout=msa_dropout,
                stochastic_depth=stochastic_depth,
                causal=False,
                share_kv=tranformer_share_kv,
                max_subtract=tranformer_max_subtract,
                d_model_scale=tranformer_d_model_scale,
                log_length_scale=tranformer_log_length_scale,
                quiet_attention=tranformer_quiet_attention,
                linear_module=linear_module,
                bos_tokens=transformer_bos_tokens,
            )
        else:
            self.transformer = nn.Identity()

        self.encoder = nn.Sequential(
            *[
                nn.BatchNorm2d(image_channels) if batch_norm else nn.Identity(),
                self.conv,
                self.pool,
                self.transformer,
            ]
        )

    def forward(self, x):
        return self.encoder(x)


class CCT(nn.Module):
    """
    Based on the Compact Convolutional Transformer (CCT) of [Hasani et al. (2021)
        *''Escaping the Big Data Paradigm with Compact Transformers''*](
        https://arxiv.org/abs/2104.05704). It's a convolutional neural network
        leading into a transformer encoder, followed by a sequence pooling layer.
    """

    def __init__(
        self,
        image_size=32,
        conv_kernel_size=3,  # Only 2 is supported for eigenvector initialisation
        pooling_type="maxpool",
        pooling_kernel_size=3,
        pooling_kernel_stride=2,
        pooling_kernel_padding=1,
        transformer_position_embedding="absolute",  # absolute or relative
        transformer_embedding_size=256,
        transformer_layers=7,
        transformer_heads=4,
        transformer_mlp_ratio=2,
        transformer_bos_tokens=4,
        tranformer_share_kv=True,
        tranformer_max_subtract=True,
        tranformer_d_model_scale=True,
        tranformer_log_length_scale=True,
        tranformer_quiet_attention=True,
        cnn_activation: nn.Module = nn.ReLU,
        cnn_activation_kwargs: Optional[dict] = None,
        transformer_activation: nn.Module = nn.GELU,
        transformer_activation_kwargs: Optional[dict] = None,
        mlp_dropout=0.0,  # The original paper got best performance from mlp_dropout=0.
        msa_dropout=0.1,  # "" msa_dropout=0.1
        stochastic_depth=0.1,  # "" stochastic_depth=0.1
        image_classes=100,
        linear_module=nn.Linear,
        image_channels=3,
        batch_norm=False,
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

        self.encoder = CCTEncoder(
            image_size=image_size,
            conv_kernel_size=conv_kernel_size,
            conv_pooling_type=pooling_type,
            conv_pooling_kernel_size=pooling_kernel_size,
            conv_pooling_kernel_stride=pooling_kernel_stride,
            conv_pooling_kernel_padding=pooling_kernel_padding,
            transformer_position_embedding=transformer_position_embedding,
            transformer_embedding_size=transformer_embedding_size,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_mlp_ratio=transformer_mlp_ratio,
            transformer_bos_tokens=transformer_bos_tokens,
            tranformer_share_kv=tranformer_share_kv,
            tranformer_max_subtract=tranformer_max_subtract,
            tranformer_d_model_scale=tranformer_d_model_scale,
            tranformer_log_length_scale=tranformer_log_length_scale,
            tranformer_quiet_attention=tranformer_quiet_attention,
            cnn_activation=cnn_activation,
            cnn_activation_kwargs=cnn_activation_kwargs,
            transformer_activation=transformer_activation,
            transformer_activation_kwargs=transformer_activation_kwargs,
            mlp_dropout=mlp_dropout,
            msa_dropout=msa_dropout,
            stochastic_depth=stochastic_depth,
            linear_module=linear_module,
            image_channels=image_channels,
            batch_norm=batch_norm,
        )
        self.pool = SequencePool(
            transformer_embedding_size, linear_module, image_classes
        )

    @property
    def sequence_length(self):
        return self.encoder.sequence_length

    def forward(self, x):
        return self.pool(self.encoder(x))
