from typing import Dict
from torch import Tensor

from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import JaccardLoss

__ALL__ = ["get_loss"]
KEY = "LOSS"


def get_loss(cfg: OmegaConf) -> nn.Module:
    args = dict(cfg[KEY].ARGS)
    args = {str(k).lower(): v for k, v in args.items()}
    loss_fn = eval(cfg[KEY].VERSION)(**args)
    return loss_fn


class MSE(nn.Module):
    def __init__(self, **kwargs):
        super(MSE, self).__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:

        loss = F.mse_loss(pred[:, :] * target[:, :, 3:], target[:, :, :3])

        return loss


class BCE(nn.Module):
    def __init__(self, **kwargs):
        super(BCE, self).__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = F.binary_cross_entropy_with_logits(pred.reshape(-1), target.reshape(-1))
        return loss


class HybridV2(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super(HybridV2, self).__init__()
        assert 0 <= alpha <= 1, "alpha should between 0 and 1"
        self.alpha = alpha

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:

        # pred  N, 12, 256, 3
        # target N, 2, 256, 3

        pred_score = pred[:, 0]  # N, 256, 3
        target_socre = target[:, 0]  # N, 256, 3

        pred_offset = pred[:, 1:]  # N, 11, 256, 3
        target_offset = target[:, 1]  # N,  256, 3
        target_offset = target_offset.type(torch.int64)

        loss = 0

        loss += (1 - self.alpha) * F.binary_cross_entropy_with_logits(
            pred_score.reshape(-1), target_socre.reshape(-1)
        )

        loss += self.alpha * F.cross_entropy(
            pred_offset, target_offset, reduction="mean"
        )

        return loss


class HybridV3(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super(HybridV3, self).__init__()
        assert 0 <= alpha <= 1, "alpha should between 0 and 1"
        self.alpha = alpha

    def forward(
        self,
        pred_point: Tensor,
        pred_offset: Tensor,
        target_point: Tensor,
        target_offset: Tensor,
    ) -> Tensor:

        # pred_point N, 1, 256, 1
        # pred_offset N, 9, 256, 2
        # target_point N, 256, 1
        # target_offset N, 256, 2

        loss = (1 - self.alpha) * F.binary_cross_entropy_with_logits(
            pred_point.reshape(-1), target_point.reshape(-1), reduction="mean"
        ) + self.alpha * F.cross_entropy(
            pred_offset, target_offset, reduction="mean", ignore_index=-999
        )

        return loss


# JaccardLoss and CE loss
class HybridV4(nn.Module):
    def __init__(self, alpha: float = 0.5, **kwargs):
        super(HybridV4, self).__init__()
        self.alpha = alpha
        self.jaccard = JaccardLoss(mode="binary")
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, pred, target, label) -> Tensor:
        losses = self.alpha * self.jaccard(pred[0], target) + (
            1 - self.alpha
        ) * self.ce(pred[1], label)
        return losses


class HybridV5(nn.Module):
    def __init__(self, alpha: float = 0.5, **kwargs):
        super(HybridV5, self).__init__()
        self.alpha = alpha
        self.jaccard = JaccardLoss(mode="binary")
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, pred, target, label) -> Tensor:

        losses = self.alpha * F.binary_cross_entropy_with_logits(
            pred[0].reshape(-1), target.reshape(-1)
        ) + (1 - self.alpha) * self.ce(pred[1], label)
        return losses


class ReduceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ReduceLoss, self).__init__()
        self.drop_loss = ["loss_classifier", "loss_objectness"]

    def forward(self, loss_dict) -> Tensor:

        for d in self.drop_loss:
            if d in loss_dict:
                loss_dict.pop(d)
        losses = sum(loss for loss in loss_dict.values())
        return losses
