import pickle
import cv2
import pandas as pd

import torch
import glob
import numpy as np
from utils.transform import get_transform
from .cvat_parser import CVATImageAnnotation

from PIL import Image
from typing import Dict  # , Tuple, List, Optional
from torch import Tensor
from torch.utils.data import Dataset
from omegaconf import OmegaConf

import xml.etree.ElementTree as ET
import sympy as sy
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm

__ALL__ = ["get_dataset"]
KEY = "DATASET"


def get_dataset(cfg: OmegaConf, split: str) -> Dataset:
    dataset = eval(cfg[KEY].VERSION)
    dataset = dataset(cfg, split)
    return dataset


def to_tensor(img) -> Tensor:
    img = torch.tensor(np.array(img), dtype=torch.float)
    img /= 255
    return img


def get_line_adjust(arr, width: int = 128):
    line = sy.Line(*arr)
    l = float(line.intersection(sy.Line([0, 0], [0, 1]))[0][1])
    m = float(line.intersection(sy.Line([width // 2, 0], [width // 2, 1]))[0][1])
    r = float(line.intersection(sy.Line([width, 0], [width, 1]))[0][1])
    arr = np.round(np.array([l, m, r]).astype(np.float64), 2)
    return arr


def get_circle_adjust(arr: list, width: int = 128):
    circle = sy.Circle(*arr)

    l = circle.intersection(sy.Line([0, 0], [0, 1]))
    l_d = [p.distance(arr[0]) for p in l]
    l = float(l[bool(l_d[0] > l_d[1])][1])

    m = circle.intersection(sy.Line([width // 2, 0], [width // 2, 1]))
    m_d = [p.distance(arr[1]) for p in m]
    m = float(m[bool(m_d[0] > m_d[1])][1])

    r = circle.intersection(sy.Line([width, 0], [width, 1]))
    r_d = [p.distance(arr[2]) for p in r]
    r = float(r[bool(r_d[0] > r_d[1])][1])

    arr = np.round(np.array([l, m, r]).astype(np.float64), 2)
    return arr


def adjust_labels(arr: list, width: int = 128):

    # arr L,W

    # define a line

    arr.sort(key=lambda x: x[0])

    if len(arr) < 2:
        raise "length of label is lese than two"

    if len(arr) == 2:
        return get_line_adjust(arr, width)

    elif len(arr) == 3:
        return get_circle_adjust(arr, width)

    else:
        arr = [arr[0], arr[-1]]
        return get_line_adjust(arr)


# TODO speed is slow, might switch to multiprocessing
def get_labels(cfg: OmegaConf) -> dict:
    labels = {}
    labels["Bark Edge"] = {}
    labels["Sapwood Edge"] = {}
    width = cfg[KEY].IMAGE_SHAPE[1]
    files = glob.glob(f"{cfg[KEY].PATH}/label/*")

    try:
        with open(f"{cfg[KEY].PATH}/temp-label.pkl", "rb") as f:
            labels = pickle.load(f)
    except:
        pass

    for f in files:
        if f.split(".")[-1] != "xml":
            continue
        doc = ET.parse(f)
        root = doc.getroot()
        for child in tqdm(root):
            if child.tag == "image":
                image = CVATImageAnnotation(child)
                filename = str(image.file.name)
                # species[filename] = sample[int(filename.split('-')[0])]
                if filename not in labels:
                    try:

                        annotation = image.get_dict()
                        label = [
                            adjust_labels(x, width).reshape(1, -1)
                            for x in annotation["Ring"]
                        ]
                        label.sort(key=lambda x: x[0][0])
                        label = np.concatenate(label)
                        labels[filename] = label

                        if "Bark Edge" in annotation:
                            labels["Bark Edge"][filename] = adjust_labels(
                                annotation["Bark Edge"][0], width
                            ).reshape(1, -1)

                        if "Sapwood Edge" in annotation:
                            labels["Sapwood Edge"][filename] = adjust_labels(
                                annotation["Sapwood Edge"][0], width
                            ).reshape(1, -1)

                    except:
                        pass

    with open(f"{cfg[KEY].PATH}/temp-label.pkl", "wb") as f:
        pickle.dump(labels, f)

    return labels


class BaseDataset(Dataset):
    r"""BaseDataset

    Args:
        cfg (OmegaConf): global config file
    """

    def __init__(self, cfg: OmegaConf, split: str) -> None:
        assert split in ["train", "test", "val"]
        self.fold = cfg["EXPERIMENT"].FOLD
        self.total_folds = cfg["EXPERIMENT"].TOTAL_FOLDS
        self.split = split
        self.labels = get_labels(cfg)
        self.cfg = cfg
        self.images = [x.split("/")[-1] for x in glob.glob(f"{cfg[KEY].PATH}/image/*")]
        self.transform = get_transform(cfg, split)
        self.return_infos = cfg[KEY].RETURN_INFOS

        sample = pd.read_csv(f"{cfg[KEY].PATH}/label/sample.csv", header=None)
        sample = sample.set_index(0)[1].to_dict()
        self.sample = sample

        if split != "test":
            self.images = [x for x in self.images if x in self.labels]

            # group = [species[x] for x in self.images]

            kf = StratifiedKFold(
                n_splits=self.total_folds, random_state=cfg.SEED, shuffle=True
            )

            index = list(kf.split(list(sample.keys()), list(sample.values())))[
                self.fold
            ][[0, 1][split == "val"]]

            index += 1

            self.images = [
                img for img in self.images if int(img.split("-")[0]) in index
            ]

        else:
            self.images = [x for x in self.images if x not in self.labels]

    def get_data_by_filename(self, filename: str):
        item = self.images.index(filename)
        return self[item]

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        filename = self.images[item]
        image = Image.open(f"{self.cfg[KEY].PATH}/image/{filename}")
        image = self.transform(image)

        channel, length, height = image.shape
        max_length = self.cfg[KEY].IMAGE_SHAPE[0]

        start_point = 0
        if length > max_length:
            if self.split != "test":
                start_point = np.random.randint(0, image.shape[1] - max_length)
            image = image[:, start_point : start_point + max_length]
        else:
            new_image = torch.ones([channel, max_length, height], dtype=image.dtype)
            new_image[:, :length, :] = image
            image = new_image

        if self.split != "test":

            label = self.labels[filename].copy()

            label = label[
                np.bitwise_and(
                    label[:, 0] >= start_point, label[:, 0] < start_point + max_length
                )
            ]
            label -= start_point

            label = np.concatenate(
                [label, np.ones(len(label), dtype=label.dtype).reshape(-1, 1)], axis=1
            )

            max_num_rings = self.cfg[KEY].LABEL_SHAPE[0]

            if len(label) <= max_num_rings:
                label = np.concatenate(
                    [
                        label,
                        np.zeros(
                            [max_num_rings - len(label), label.shape[1]],
                            dtype=label.dtype,
                        ),
                    ]
                )
            else:
                label = label[:max_num_rings]
            label = torch.tensor(label)
            return {"image": image, "label": label}

        return {"image": image, "start_point": start_point, "filename": filename}

    def __len__(self) -> int:
        return len(self.images)


class DatasetV2(BaseDataset):
    def __getitem__(self, item: int) -> Dict[str, Tensor]:

        filename = self.images[item]
        image = Image.open(f"{self.cfg[KEY].PATH}/image/{filename}")
        image = self.transform(image)

        channel, length, height = image.shape
        max_length = self.cfg[KEY].IMAGE_SHAPE[0]

        start_point = 0

        if length > max_length:
            # if self.split != "test":
            if self.split != "train":
                start_point = np.random.randint(0, image.shape[1] - max_length)
            image = image[:, start_point : start_point + max_length]
        else:
            new_image = torch.ones([channel, max_length, height], dtype=image.dtype)
            new_image[:, :length, :] = image
            image = new_image

        if self.split != "test":
            label = self.labels[filename].copy()
            mask = np.zeros(self.cfg[KEY].LABEL_SHAPE)

            filter = True
            for i in range(3):
                filter = filter & np.bitwise_and(
                    label[:, i] >= start_point, label[:, i] < start_point + max_length
                )
            label = label[filter]

            if len(label) > 0:
                label -= start_point
                label //= self.cfg[KEY].IMAGE_SHAPE[0] // self.cfg[KEY].LABEL_SHAPE[0]
                label = label.astype(int)

                for i in range(3):
                    mask[label[:, i], i] = 1

                new_mask = cv2.GaussianBlur(mask, (1, self.cfg[KEY].SMOOTH), 0)
                mask = new_mask / np.median(new_mask[mask == 1])
                mask = np.clip(mask, 0, 1)

            mask = torch.tensor(mask, dtype=torch.float)

            if self.return_infos:
                return {
                    "image": image,
                    "label": mask,
                    "start_point": start_point,
                    "filename": filename,
                }
            else:
                return {"image": image, "label": mask}

        return {"image": image, "start_point": start_point, "filename": filename}


class DatasetV3(BaseDataset):
    def __getitem__(self, item: int) -> Dict[str, Tensor]:

        filename = self.images[item]
        image = Image.open(f"{self.cfg[KEY].PATH}/image/{filename}")
        image = self.transform(image)

        channel, length, height = image.shape
        max_length, max_width = self.cfg[KEY].IMAGE_SHAPE

        start_point = 0

        if length > max_length:
            if self.split != "test":
                start_point = np.random.randint(0, image.shape[1] - max_length)
            image = image[:, start_point : start_point + max_length]
        else:
            new_image = torch.ones([channel, max_length, height], dtype=image.dtype)
            new_image[:, :length, :] = image
            image = new_image

        if self.split != "test":
            label = self.labels[filename].copy()
            filter = True
            for i in range(3):
                filter = filter & np.bitwise_and(
                    label[:, i] >= start_point, label[:, i] < start_point + max_length
                )
            label = label[filter]  # N,3

            if len(label) == 0:
                start_point = 0
                label = self.labels[filename].copy()
                filter = True
                for i in range(3):
                    filter = filter & np.bitwise_and(
                        label[:, i] >= start_point,
                        label[:, i] < start_point + max_length,
                    )
                label = label[filter]

            label -= start_point
            new_labels = np.zeros([len(label) + 2, 3])
            new_labels[1:-1] = label
            new_labels[-1] = max_length
            diff = new_labels[1:] - new_labels[:-1]
            low = np.clip(
                (new_labels[1:-1] - 1 / 2 * diff[:-1]).min(1), 0, max_length
            ).reshape(-1, 1)
            up = np.clip(
                (new_labels[1:-1] + 1 / 2 * diff[1:]).max(1), 0, max_length
            ).reshape(-1, 1)
            boxes = np.concatenate(
                [
                    low,
                    np.zeros(len(low)).reshape(-1, 1),
                    up,
                    np.ones(len(up)).reshape(-1, 1) * max_width,
                ],
                axis=1,
            )
            boxes = torch.tensor(boxes, dtype=torch.float)

            x = label.reshape(-1, 3, 1)
            y = np.concatenate(
                [
                    np.zeros(len(x)).reshape(-1, 1, 1),
                    np.ones(len(x)).reshape(-1, 1, 1) * max_width / 2,
                    np.ones(len(x)).reshape(-1, 1, 1) * max_width,
                ],
                axis=1,
            )
            z = np.ones_like(x)
            keypoints = np.concatenate([x, y, z], axis=2)
            keypoints = torch.tensor(keypoints, dtype=boxes.dtype)
            return {
                "image": image,
                "target": {
                    "boxes": boxes,
                    "keypoints": keypoints,
                    "labels": torch.ones(len(keypoints), dtype=torch.int64),
                },
            }

        return {"image": image, "start_point": start_point, "filename": filename}


class DatasetV4(BaseDataset):
    def __getitem__(self, item: int) -> Dict[str, Tensor]:

        filename = self.images[item]
        image = Image.open(f"{self.cfg[KEY].PATH}/image/{filename}")
        image = self.transform(image)

        channel, length, height = image.shape
        max_length = self.cfg[KEY].IMAGE_SHAPE[0]

        start_point = 0
        if length > max_length:
            if self.split != "test":
                start_point = np.random.randint(0, image.shape[1] - max_length)
            image = image[:, start_point : start_point + max_length]
        else:
            new_image = torch.ones([channel, max_length, height], dtype=image.dtype)
            new_image[:, :length, :] = image
            image = new_image

        if self.split != "test":
            label = self.labels[filename].copy()
            mask = np.zeros(self.cfg[KEY].LABEL_SHAPE)

            filter = True
            for i in range(3):
                filter = filter & np.bitwise_and(
                    label[:, i] >= start_point, label[:, i] < start_point + max_length
                )
            label = label[filter]

            if len(label) > 0:
                label -= start_point
                label //= self.cfg[KEY].IMAGE_SHAPE[0] // self.cfg[KEY].LABEL_SHAPE[0]
                label = label.astype(int)

                for i in range(3):
                    mask[label[:, i], i] = 1

                new_mask = cv2.GaussianBlur(mask, (1, self.cfg[KEY].SMOOTH), 0)
                mask = new_mask / np.median(new_mask[mask == 1])
                mask = np.clip(mask, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float)

            offset_mask = np.zeros([len(mask), 3])

            if len(label) > 0:
                offset_threshold = self.cfg[KEY].OFFSET_THRESHOLD
                offset = np.zeros([len(label), 3])
                for i in range(3):
                    offset[:, i] = (
                        np.clip(
                            label[:, (i + 1) % 3] - label[:, i],
                            -offset_threshold,
                            offset_threshold,
                        )
                        + offset_threshold
                        + 1
                    )

                for i in range(3):
                    for j in range(self.cfg[KEY].SMOOTH):
                        offset_mask[
                            np.clip(
                                label[:, i] + j - self.cfg[KEY].SMOOTH // 2,
                                0,
                                len(mask) - 1,
                            ),
                            i,
                        ] = offset[:, i]

            offset_mask = torch.tensor(offset_mask, dtype=torch.float)
            mask = torch.stack([mask, offset_mask], 0)

            return {"image": image, "label": mask}

        return {"image": image, "start_point": start_point, "filename": filename}


class DatasetV5(BaseDataset):
    def __getitem__(self, item: int) -> Dict[str, Tensor]:

        filename = self.images[item]
        image = Image.open(f"{self.cfg[KEY].PATH}/image/{filename}")
        image = self.transform(image)

        channel, length, height = image.shape
        max_length = self.cfg[KEY].IMAGE_SHAPE[0]

        start_point = 0
        if length > max_length:
            if self.split != "test":
                start_point = np.random.randint(0, image.shape[1] - max_length)
            image = image[:, start_point : start_point + max_length]
        else:
            new_image = torch.ones([channel, max_length, height], dtype=image.dtype)
            new_image[:, :length, :] = image
            image = new_image

        if self.split != "test":
            label = self.labels[filename].copy()
            mask = np.zeros(self.cfg[KEY].LABEL_SHAPE)

            filter = True
            for i in range(3):
                filter = filter & np.bitwise_and(
                    label[:, i] >= start_point, label[:, i] < start_point + max_length
                )
            label = label[filter]

            if len(label) > 0:
                label -= start_point
                label //= self.cfg[KEY].IMAGE_SHAPE[0] // self.cfg[KEY].LABEL_SHAPE[0]
                label = label.astype(int)
                mask[label[:, 1], 0] = 1
                new_mask = cv2.GaussianBlur(mask, (1, self.cfg[KEY].SMOOTH), 0)
                mask = new_mask / np.median(new_mask[mask == 1])
                mask = np.clip(mask, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float)

            offset_mask = np.ones([len(mask), 2]) * -999

            if len(label) > 0:
                offset_threshold = self.cfg[KEY].OFFSET_THRESHOLD
                offset = np.zeros([len(label), 2])

                offset[:, 0] = (
                    np.clip(
                        label[:, 1] - label[:, 0],
                        -offset_threshold,
                        offset_threshold,
                    )
                    + offset_threshold
                )

                offset[:, 1] = (
                    np.clip(
                        label[:, 1] - label[:, 2],
                        -offset_threshold,
                        offset_threshold,
                    )
                    + offset_threshold
                )

                for i in range(2):
                    for j in range(self.cfg[KEY].SMOOTH):
                        offset_mask[
                            np.clip(
                                label[:, i] + j - self.cfg[KEY].SMOOTH // 2,
                                0,
                                len(mask) - 1,
                            ),
                            i,
                        ] = offset[:, i]

            offset_mask = torch.tensor(offset_mask, dtype=torch.int64)
            # mask = torch.stack([mask, offset_mask], 0)

            return {"image": image, "mask": mask, "offset_mask": offset_mask}

        return {"image": image, "start_point": start_point, "filename": filename}


class DatasetV6(BaseDataset):
    def __getitem__(self, item: int) -> Dict[str, Tensor]:

        filename = self.images[item]
        image = Image.open(f"{self.cfg[KEY].PATH}/image/{filename}")
        image = self.transform(image)

        channel, length, height = image.shape
        max_length = self.cfg[KEY].IMAGE_SHAPE[0]

        start_point = 0
        if length > max_length:
            if self.split != "test":
                start_point = np.random.randint(0, image.shape[1] - max_length)
            image = image[:, start_point : start_point + max_length]
        else:
            new_image = torch.ones([channel, max_length, height], dtype=image.dtype)
            new_image[:, :length, :] = image
            image = new_image

        if self.split != "test":
            label = self.labels[filename].copy()
            mask = np.zeros(self.cfg[KEY].LABEL_SHAPE)

            filter = True
            for i in range(3):
                filter = filter & np.bitwise_and(
                    label[:, i] >= start_point, label[:, i] < start_point + max_length
                )
            label = label[filter]

            if len(label) > 0:
                label -= start_point
                label //= self.cfg[KEY].IMAGE_SHAPE[0] // self.cfg[KEY].LABEL_SHAPE[0]
                label = label.astype(int)
                mask[label[:, 1], 0] = 1
                new_mask = cv2.GaussianBlur(mask, (1, self.cfg[KEY].SMOOTH), 0)
                mask = new_mask / np.median(new_mask[mask == 1])
                mask = np.clip(mask, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float)

            offset_mask = np.ones([len(mask), 2]) * -999

            if len(label) > 0:
                offset_threshold = self.cfg[KEY].OFFSET_THRESHOLD
                offset = np.zeros([len(label), 2])

                offset[:, 0] = (
                    np.clip(
                        label[:, 1] - label[:, 0],
                        -offset_threshold,
                        offset_threshold,
                    )
                    + offset_threshold
                )

                offset[:, 1] = (
                    np.clip(
                        label[:, 1] - label[:, 2],
                        -offset_threshold,
                        offset_threshold,
                    )
                    + offset_threshold
                )

                for i in range(2):
                    for j in range(self.cfg[KEY].SMOOTH):
                        offset_mask[
                            np.clip(
                                label[:, i] + j - self.cfg[KEY].SMOOTH // 2,
                                0,
                                len(mask) - 1,
                            ),
                            i,
                        ] = offset[:, i]

            offset_mask = torch.tensor(offset_mask, dtype=torch.int64)

            return {"image": image, "mask": mask, "offset_mask": offset_mask}

        return {"image": image, "start_point": start_point, "filename": filename}


class DatasetV8(Dataset):
    r"""DatasetV7

    Args:
        cfg (OmegaConf): global config file
    """

    def __init__(self, cfg: OmegaConf, split: str) -> None:
        assert split in ["train", "test", "val"]
        self.fold = cfg["EXPERIMENT"].FOLD
        self.total_folds = cfg["EXPERIMENT"].TOTAL_FOLDS
        self.split = split
        self.cfg = cfg

        self.images = [x.split("/")[-1] for x in glob.glob(f"{cfg[KEY].PATH}/image/*")]
        self.labels = [x.split("/")[-1] for x in glob.glob(f"{cfg[KEY].PATH}/mask/*")]

        self.images = [x for x in self.images if x in self.labels]

        self.transform = get_transform(cfg, split)
        self.return_infos = cfg[KEY].RETURN_INFOS

        sample = pd.read_csv(f"{cfg[KEY].PATH}/label/sample.csv", header=None)
        sample = sample.set_index(0)[1].to_dict()
        self.sample = sample

        kf = StratifiedKFold(
            n_splits=self.total_folds, random_state=cfg.SEED, shuffle=True
        )

        index = list(kf.split(list(sample.keys()), list(sample.values())))[self.fold][
            [1, 0][split == "train"]
        ]

        index += 1

        self.images = [img for img in self.images if int(img.split("-")[0]) in index]

        species2label = list(set(list(self.sample.values())))
        species2label = dict(zip(species2label, range(len(species2label))))
        self.species2label = species2label

    def get_data_by_filename(self, filename: str):
        item = self.images.index(filename)
        return self[item]

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        filename = self.images[item]
        image = Image.open(f"{self.cfg[KEY].PATH}/image/{filename}")
        mask = torch.tensor(
            (np.array(Image.open(f"{self.cfg[KEY].PATH}/mask/{filename}")) != 0).astype(
                float
            )
        )[None]

        image = self.transform(image)
        max_length = self.cfg[KEY].IMAGE_SHAPE[0]

        if self.split != "test":
            start_point = np.random.randint(0, image.shape[1] - max_length)
            image = image[:, start_point : start_point + max_length]
            mask = mask[:, start_point : start_point + max_length]
            return {"image": image, "label": mask}
        else:
            return {"image": image, "label": mask, "filename": filename}

    def __len__(self) -> int:
        return len(self.images)


class DatasetV9(DatasetV8):
    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        filename = self.images[item]
        image = Image.open(f"{self.cfg[KEY].PATH}/image/{filename}")

        label = self.species2label[self.sample[int(filename.split("-")[0])]]
        label = torch.tensor(label)

        mask = torch.tensor(
            (np.array(Image.open(f"{self.cfg[KEY].PATH}/mask/{filename}")) != 0).astype(
                float
            )
        )[None]

        ignore = torch.tensor(
            (
                np.array(Image.open(f"{self.cfg[KEY].PATH}/ignore/{filename}")) != 0
            ).astype(float)
        )[None]

        image = self.transform(image)
        max_length = self.cfg[KEY].IMAGE_SHAPE[0]

        if self.split != "test":
            r = torch.where(ignore[0, :, 0] == 1)[0].numpy()
            start_point = np.random.choice(r)
            start_point = min(start_point, image.shape[1] - max_length)

            image = image[:, start_point : start_point + max_length]
            mask = mask[:, start_point : start_point + max_length]
            return {"image": image, "mask": mask, "label": label}

        else:
            return {
                "image": image,
                "label": mask,
                "filename": filename,
                "ignore": ignore,
            }
