from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torchmetrics.metric import Metric

from .mean_ap import eval_map

__ALL__ = ["get_metric"]
KEY = "METRIC"


def get_metric(cfg: OmegaConf) -> nn.Module:
    args = dict(cfg[KEY].ARGS)
    args = {str(k).lower(): v for k, v in args.items()}
    loss_fn = eval(cfg[KEY].VERSION)(**args)
    return loss_fn


class Base(Metric):
    def __init__(self, dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "loss", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )

        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )

    def update(self, loss: torch.Tensor):
        self.loss += loss
        self.total += 1
        return loss

    def compute(self):
        return self.loss / self.total


def target_transform(targets):
    new_targets = []
    for target in targets:
        new_targets.append(
            {
                "bboxes": target["boxes"].cpu().detach().numpy(),
                "labels": target["labels"].cpu().detach().numpy(),
            }
        )
    return new_targets


def predict_transform(predicts):
    new_predicts = []
    for pred in predicts:
        b = []
        for label in range(2):
            b += [pred["boxes"][pred["labels"] == label].detach().cpu().numpy()]
        new_predicts.append(b)
    return new_predicts


class mAP(Metric):
    def __init__(self, dist_sync_on_step=False, threshold=0.5, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "loss", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )

        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )

        self.threshold = threshold

    def update(self, predict: List[Dict], target):
        predict = predict_transform(predict)
        target = target_transform(target)
        loss = -1 * torch.tensor(
            eval_map(predict, target, iou_thr=self.threshold)[0], dtype=torch.float
        )
        self.loss += loss
        return loss

    def compute(self):
        return self.loss / self.total
