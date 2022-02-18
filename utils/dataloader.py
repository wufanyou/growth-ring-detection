# Created by fw at 12/31/20
from typing import List, Dict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from utils.dataset import get_dataset
import torch

__ALL__ = ["get_dataloader"]
KEY = "DATALOADER"


def object_detection_collate(batch: List[Dict]) -> List:
    # print(batch)
    data = [item["image"] for item in batch]
    target = [item["target"] for item in batch]
    # target = torch.LongTensor(target)
    return data, target


def get_dataloader(cfg: OmegaConf) -> LightningDataModule:
    dataloader = eval(cfg[KEY].VERSION)(cfg=cfg)
    return dataloader


class BaseDataLoader(LightningDataModule):
    r"""BaseDataLoader

    Args:
        cfg (OmegaConf): global config file
    """

    def __init__(self, cfg: OmegaConf):
        super(BaseDataLoader, self).__init__()
        self.cfg = cfg
        self.dims = cfg[KEY].DIMS
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage=None) -> None:

        if stage == "fit" or stage is None:
            self.val_dataset = get_dataset(self.cfg, "val")

            self.train_dataset = get_dataset(self.cfg, "train")

        if stage == "test" or stage is None:
            self.test_dataset = get_dataset(self.cfg, "test")

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test")

    def get_dataloader(self, split: str) -> DataLoader:
        assert split in ["train", "test", "val"]
        batch_size = self.cfg[KEY].BATCH_SIZE[split.upper()]
        num_workers = self.cfg[KEY].NUM_WORKERS

        dataset = eval(f"self.{split}_dataset")

        collate_fn = None
        if self.cfg[KEY].get("COLLATE_FN"):
            collate_fn = eval(self.cfg[KEY].get("COLLATE_FN"))

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=split == "train",
            pin_memory=True,
            collate_fn=collate_fn,
        )

        return loader


class DataLoaderV2(BaseDataLoader):
    def get_dataloader(self, split: str) -> DataLoader:
        assert split in ["train", "test", "val"]
        batch_size = self.cfg[KEY].BATCH_SIZE[split.upper()]
        num_workers = self.cfg[KEY].NUM_WORKERS

        dataset = eval(f"self.{split}_dataset")

        collate_fn = None
        if self.cfg[KEY].get("COLLATE_FN"):
            collate_fn = eval(self.cfg[KEY].get("COLLATE_FN"))

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=split == "train",
            pin_memory=False,  # For object detection
            collate_fn=collate_fn,
        )

        return loader
