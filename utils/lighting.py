# Created by fw at 12/31/20

from abc import ABC
from pytorch_lightning import LightningModule
from omegaconf import OmegaConf
from utils.model import get_model
from utils.optimizer import get_optimizer
from utils.losses import get_loss
from utils.metrics import get_metric
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple, Union, List, Optional
from torchvision.models.detection.roi_heads import keypointrcnn_loss

__ALL__ = ["get_lighting"]
KEY = "LIGHTING"


def get_lighting(cfg: OmegaConf) -> LightningModule:
    model = eval(cfg[KEY].VERSION)(cfg)
    return model


class BaseLightingModule(LightningModule):
    r"""BaseLightingModule.

    Args:
        cfg (OmegaConf): global config file
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(cfg)
        self.metrics = get_metric(cfg)
        self.loss_fn = get_loss(cfg)

    def forward(self, image) -> Tensor:
        output = self.model(image)
        return output

    def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
        y_hat = self(batch["image"])
        loss = self.loss_fn(y_hat, batch["label"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: Tensor) -> None:
        y_hat = self(batch["image"])
        loss = self.loss_fn(y_hat, batch["label"])
        loss = self.metrics(loss)
        return loss

    def validation_epoch_end(self, outputs: list) -> None:
        self.log("loss", self.metrics.compute())

    def configure_optimizers(
        self,
    ) -> Union[Tuple[List[Optimizer], Union[List[LambdaLR], List[dict]]], Optimizer]:
        optimizer, scheduler = get_optimizer(self.cfg, self.model)
        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return optimizer


class LightingModuleV2(BaseLightingModule):
    def forward(self, image: List[Tensor], target=None) -> Tensor:
        output = self.model(image, target)
        return output

    def training_step(self, batch: List, batch_idx: Tensor) -> Tensor:
        image, target = batch
        loss = self.loss_fn(self(image, target))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: List, batch_idx: Tensor) -> Tensor:
        image, target = batch
        score = self.metrics(self(image), target)
        return score


class LightingModuleV3(BaseLightingModule):
    def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
        mid_point, offset_bin = self(batch["image"])
        loss = self.loss_fn(mid_point, offset_bin, batch["mask"], batch["offset_mask"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: Tensor) -> None:
        mid_point, offset_bin = self(batch["image"])
        loss = self.loss_fn(mid_point, offset_bin, batch["mask"], batch["offset_mask"])
        loss = self.metrics(loss)
        return loss


class LightingModuleV4(BaseLightingModule):
    def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
        y_hat = self(batch["image"])
        loss = self.loss_fn(y_hat, batch["mask"], batch["label"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: Tensor) -> None:
        y_hat = self(batch["image"])
        loss = self.loss_fn(y_hat, batch["mask"], batch["label"])
        loss = self.metrics(loss)
        return loss
