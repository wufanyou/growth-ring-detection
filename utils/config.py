# Created by fw at 12/30/20
from omegaconf import OmegaConf
from typing import Optional

__ALL__ = ["get_cfg"]
KEY = "CONFIG"


def get_filename(path: str) -> str:
    filename = path.split("/")[-1].split(".")[0]
    return filename


def get_cfg(
    path: Optional[str] = None, fold: int = 0, total_folds: int = 5
) -> OmegaConf:
    if path is not None:
        cfg = OmegaConf.load(path)
        cfg = OmegaConf.merge(_C, cfg)
        cfg.EXPERIMENT.NAME = get_filename(path)
        cfg.EXPERIMENT.FOLD = fold
        cfg.EXPERIMENT.TOTAL_FOLDS = total_folds
    else:
        cfg = _C.copy()
        cfg.EXPERIMENT.NAME = "NA"
    return cfg


# experiment
_C = OmegaConf.create()
_C.SEED = 2021

_C.EXPERIMENT = OmegaConf.create()
_C.EXPERIMENT.NAME = ""
_C.EXPERIMENT.FOLD = 0
_C.EXPERIMENT.TOTAL_FOLDS = 5
_C.EXPERIMENT.LOGGER = OmegaConf.create()
_C.EXPERIMENT.LOGGER.VERSION = "MLFlowLogger"
_C.EXPERIMENT.LOGGER.TRACKING_URI = "file:./mlruns"
_C.EXPERIMENT.LOGGER.EXPERIMENT_NAME = "Default"

_C.EXPERIMENT.SAVER = OmegaConf.create()
_C.EXPERIMENT.SAVER.MONITOR = "loss"
_C.EXPERIMENT.SAVER.VERBOSE = False
_C.EXPERIMENT.SAVER.DIRPATH = "./models/"
_C.EXPERIMENT.SAVER.FILENAME = "{experiment}-{{epoch}}-{{loss:.4f}}"
_C.EXPERIMENT.SAVER.SAVE_TOP_K = 1
_C.EXPERIMENT.SAVER.SAVE_WEIGHTS_ONLY = True
_C.EXPERIMENT.SAVER.MODE = "min"

# dataset
_C.DATASET = OmegaConf.create()
_C.DATASET.VERSION = "BaseDataset"
_C.DATASET.PATH = "../train-data/"
_C.DATASET.IMAGE_SHAPE = (1024, 128)
_C.DATASET.LABEL_SHAPE = (50, 4)
_C.DATASET.OFFSET_THRESHOLD = 4
_C.DATASET.RETURN_INFOS = False

# data loader
_C.DATALOADER = OmegaConf.create()
_C.DATALOADER.VERSION = "BaseDataLoader"
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.DIMS = (3, 2048, 128)
_C.DATALOADER.BATCH_SIZE = OmegaConf.create()
_C.DATALOADER.BATCH_SIZE.TRAIN = 8
_C.DATALOADER.BATCH_SIZE.TEST = 8
_C.DATALOADER.BATCH_SIZE.VAL = 8


# transform
_C.TRANSFORM = OmegaConf.create()
_C.TRANSFORM.VERSION = "TransformV2"
# _C.TRANSFORM.RESIZE = (1024, 128)


# model
_C.MODEL = OmegaConf.create()
_C.MODEL.IN_CHANNELS = 3
_C.MODEL.PRETRAINED = None
_C.MODEL.ENCODER = None
_C.MODEL.VERSION = "FPN"

# loss
_C.LOSS = OmegaConf.create()
_C.LOSS.VERSION = "MSE"
_C.LOSS.ARGS = OmegaConf.create()
# _C.LOSS.ARGS.ALPHA = 0.5


# optimizer
_C.OPTIMIZER = OmegaConf.create()
_C.OPTIMIZER.VERSION = "Adam"

_C.OPTIMIZER.ARGS = OmegaConf.create()
_C.OPTIMIZER.ARGS.LR = 0.0005

_C.OPTIMIZER.SCHEDULER = OmegaConf.create()
_C.OPTIMIZER.SCHEDULER.USE = True
_C.OPTIMIZER.SCHEDULER.VERSION = "linear_schedule_with_warmup"
_C.OPTIMIZER.SCHEDULER.ARGS = OmegaConf.create()
_C.OPTIMIZER.SCHEDULER.ARGS.NUM_WARMUP_STEPS = 1000


# lighting
_C.LIGHTING = OmegaConf.create()
_C.LIGHTING.VERSION = "BaseLightingModule"


_C.METRIC = OmegaConf.create()
_C.METRIC.VERSION = "Base"
_C.METRIC.ARGS = OmegaConf.create()
_C.METRIC.ARGS.dist_sync_on_step = False

# trainer
_C.TRAINER = OmegaConf.create()
# _C.TRAINER.MAX_STEPS = 500
_C.TRAINER.GPUS = 1
_C.TRAINER.AUTO_SELECT_GPUS = True
