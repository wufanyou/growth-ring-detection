from utils import *
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_only
import argparse
import os


@rank_zero_only
def print_cfg(cfg):
    print(cfg)


@rank_zero_only
def make_dir(cfg):
    os.makedirs(cfg.EXPERIMENT.SAVER.DIRPATH, exist_ok=True)


def main(cfg):
    model = get_lighting(cfg)
    trainer = get_trainer(cfg)
    dataloader = get_dataloader(cfg)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "-c",
        "--config",
        default="config/v2.yaml",
        type=str,
    )

    parser.add_argument(
        "-f",
        "--fold",
        default=0,
        type=int,
    )

    parser.add_argument(
        "-t",
        "--total-folds",
        default=5,
        type=int,
    )

    args = parser.parse_args()
    cfg = get_cfg(args.config, fold=args.fold, total_folds=args.total_folds)
    seed_everything(cfg.SEED)
    make_dir(cfg)
    print_cfg(cfg)
    main(cfg)
    filename = cfg['EXPERIMENT']['NAME'] + '-' + str(cfg['EXPERIMENT']['FOLD']) + '.ckpt'
    os.system(f'mv models/last.ckpt models/{filename}')
