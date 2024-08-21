# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2021 ashleve
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the pytorch lightning hydra template
# from https://github.com/ashleve/lightning-hydra-template
#################################################################################### 

import torch
import os
import pickle
import numpy as np
from typing import List, Tuple

import hydra
import pyrootutils
import lightning as L
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils, constants
from src.utils.utils import find_latest_version

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # load a checkpoint
    try:
        ckpt = torch.load(cfg.ckpt_path)
    except:
        print(f'Checkpoint: {cfg.ckpt_path} is not valid')
        last_ckpt_path = find_latest_version(cfg.last_ckpt_path)
        print(f'Loading the latest last checkpoint: {last_ckpt_path}')
        ckpt = torch.load(last_ckpt_path)

    model.load_state_dict(ckpt['state_dict'], strict=True)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule)

    with_without_weight_deacy = 'w' if cfg['model']['optimizer']['weight_decay'] > 0 else 'wo'
    save_path = os.path.join(
        cfg['paths']['log_dir'],
        'calibration',
        f"{cfg['data']['datasets']['dataset']}_{with_without_weight_deacy}_{cfg['tags'][0]}_seed{cfg.seed}.pkl")

    metric_dict = trainer.callback_metrics
    test_dataloader = datamodule.test_dataloader()

    model.eval()
    model.to('cuda')

    total_logits, total_labels = [], []
    for i, input_dict in enumerate(test_dataloader):
        output_dict = model.model_step(utils.to_device(input_dict))
        logits = output_dict[constants.LOGITS]
        labels = output_dict[constants.LABELS]

        total_logits.append(logits.detach().cpu().numpy())
        total_labels.append(labels.detach().cpu().numpy())

    total_logits = np.concatenate(total_logits)
    total_labels = np.concatenate(total_labels)

    result_dict = {'logits': total_logits, 'labels': total_labels}

    with open(save_path, 'wb') as f:
        pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
