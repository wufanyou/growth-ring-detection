from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.encoders import get_encoder
from typing import Optional
import torch.nn as nn
import torch


class CustomizeHead(nn.Sequential):
    def __init__(
        self,
        in_channels,
        in_features=512,
        output_shape=(200, 4),
        kernel_size=(3, 32),
        padding=(1, 0),
    ):
        conv2d = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=padding)
        bn = nn.BatchNorm2d(1)
        relu = nn.ReLU()
        flatten = nn.Flatten()
        linear = nn.Linear(in_features, prod(output_shape))
        unflatten = nn.Unflatten(1, unflattened_size=output_shape)
        super().__init__(conv2d, bn, relu, flatten, linear, unflatten)


class FPN(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        head_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.classification_head = None
        head_params = head_params if head_params is not None else {}
        self.segmentation_head = CustomizeHead(
            in_channels=self.decoder.out_channels, **head_params
        )
        self.name = "fpn-{}".format(encoder_name)
        self.initialize()
