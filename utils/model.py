from omegaconf import OmegaConf, ListConfig
from typing import Optional

import torch.nn as nn
import torch
from .models import *

__ALL__ = ["get_model"]
KEY = "MODEL"


def load_pretrain_model(
    model: nn.Module, pretrain: Optional[str], remove: Optional[int] = 6
) -> None:
    if pretrain is not None:
        pretrain = torch.load(pretrain, map_location="cpu")
        if "state_dict" in pretrain:
            pretrain = pretrain["state_dict"]
        weight = model.state_dict()
        for k, v in pretrain.items():
            k = k[remove:]
            if k in weight:
                if v.shape == weight[k].shape:
                    weight[k] = v
        model.load_state_dict(weight)


def replace_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.SiLU(inplace=True))
        else:
            replace_relu(child)


def get_model(cfg: OmegaConf) -> nn.Module:

    if cfg[KEY].VERSION == "keypointrcnn":
        import torchvision

        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            min_size=128, max_size=1024, num_keypoints=3, num_classes=2
        )

    else:
        try:
            head_params = dict(cfg[KEY].HEAD_PARAMS)
        except:
            head_params = {}

        for k, v in head_params.items():
            if type(v) == ListConfig:
                head_params[k] = tuple(v)

        cls = eval(cfg[KEY].VERSION)
        model = cls(
            encoder_name=cfg[KEY].ENCODER,
            in_channels=cfg[KEY].IN_CHANNELS,
            head_params=head_params,
        )

    if cfg[KEY].REPLACE_RELU:
        replace_relu(model)

    load_pretrain_model(model, cfg[KEY].PRETRAINED)
    return model
