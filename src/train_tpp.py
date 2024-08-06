# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import os
import pickle
import hydra
import lightning as L
import pyrootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
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

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

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

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    # for aux, save prediction results
    # (load dataset with exclusion indices, make predictions on them and save them)
    aux_idx, aux_num = cfg['data']['aux_idx'], cfg['data']['aux_num']
    if aux_idx >= 0:
        state_dict = torch.load(trainer.checkpoint_callback.best_model_path)['state_dict']
        model.load_state_dict(state_dict)
        model.to('cuda')
        model.eval()
        aux_infer_dataloader = datamodule.axu_infer_dataloader()

        aux_loss_dir = os.path.abspath(
            os.path.join(cfg["paths"]["output_dir"], os.pardir))
        aux_event_loss_path = os.path.join(aux_loss_dir, f"aux{aux_num}_losses.pkl")
        aux_time_pred_path = os.path.join(aux_loss_dir, f"aux{aux_num}_preds.pkl")
        aux_cls_logit_path = os.path.join(aux_loss_dir, f"aux{aux_num}_logits.pkl")
        aux_mu_path = os.path.join(aux_loss_dir, f"aux{aux_num}_mus.pkl")
        aux_sigma_path = os.path.join(aux_loss_dir, f"aux{aux_num}_sigmas.pkl")
        aux_log_weight_path = os.path.join(aux_loss_dir, f"aux{aux_num}_log_weights.pkl")

        total_event_losses = {}
        total_cls_logits = {}
        total_time_preds = {}
        total_dist_mus = {}
        total_dist_sigmas = {}
        total_log_weights = {}
        for i, input_dict in enumerate(aux_infer_dataloader):
            with torch.no_grad():
                output_dict = model.model_step(utils.to_device(input_dict))
                indices = input_dict[constants.INDICES]

                try:
                    cls_logit_dict = {
                        idx.item(): logit.detach().cpu().numpy() for idx, logit in zip(
                            indices, output_dict[constants.CLS_LOGITS])}
                except:
                    cls_logit_dict = {}

                time_pred_dict = {
                    idx.item(): time_pred.detach().cpu().numpy() for idx, time_pred in zip(
                        indices, output_dict[constants.TIME_PREDS])}

                event_ll, surv_ll = (
                    output_dict[constants.EVENT_LL], output_dict[constants.SURV_LL])
                event_losses = -(event_ll + surv_ll)

                event_loss_dict = {
                    idx.item(): loss.detach().cpu().numpy() for idx, loss in zip(
                        indices, event_losses)}

                dist_mu_dict = {
                    idx.item(): dist_mu.detach().cpu().numpy() for idx, dist_mu in zip(
                        indices, output_dict[constants.DIST_MU])}

                dist_sigma_dict = {
                    idx.item(): dist_sigma.detach().cpu().numpy() for idx, dist_sigma in zip(
                        indices, output_dict[constants.DIST_SIGMA])}

                log_weight_dict = {
                    idx.item(): log_weight.detach().cpu().numpy() for idx, log_weight in zip(
                        indices, output_dict[constants.LOG_WEIGHTS])}

                total_event_losses.update(event_loss_dict)
                total_cls_logits.update(cls_logit_dict)
                total_time_preds.update(time_pred_dict)
                total_dist_mus.update(dist_mu_dict)
                total_dist_sigmas.update(dist_sigma_dict)
                total_log_weights.update(log_weight_dict)

        if os.path.exists(aux_event_loss_path):
            if aux_idx == 0:
                os.remove(aux_event_loss_path)
                print(f"Rmoeved an existing file: {aux_event_loss_path}")
            else:
                with open(aux_event_loss_path, "rb") as f:
                    prev_event_losses = pickle.load(f)
                total_event_losses.update(prev_event_losses)

        with open(aux_event_loss_path, "wb") as f:
            pickle.dump(total_event_losses, f, -1)
        log.info(
            f"Inference aux {aux_idx+1}/{aux_num} is saved to {aux_event_loss_path}: element num = {len(total_event_losses.keys())}")

        if os.path.exists(aux_cls_logit_path):
            if aux_idx == 0:
                os.remove(aux_cls_logit_path)
                print(f"Rmoeved an existing file: {aux_cls_logit_path}")
            else:
                with open(aux_cls_logit_path, "rb") as f:
                    prev_cls_logits = pickle.load(f)
                total_cls_logits.update(prev_cls_logits)

        if total_cls_logits:
            with open(aux_cls_logit_path, "wb") as f:
                pickle.dump(total_cls_logits, f, -1)
            log.info(
                f"Inference aux {aux_idx+1}/{aux_num} is saved to {aux_cls_logit_path}: element num = {len(total_cls_logits.keys())}")

        # Mu and sigma
        if os.path.exists(aux_mu_path):
            if aux_idx == 0:
                os.remove(aux_mu_path)
                print(f"Rmoeved an existing file: {aux_mu_path}")
            else:
                with open(aux_mu_path, "rb") as f:
                    prev_mu = pickle.load(f)
                total_dist_mus.update(prev_mu)

        with open(aux_mu_path, "wb") as f:
            pickle.dump(total_dist_mus, f, -1)
        log.info(
            f"inference aux {aux_idx+1}/{aux_num} is saved to {aux_mu_path}: element num = {len(total_dist_mus.keys())}")

        if os.path.exists(aux_sigma_path):
            if aux_idx == 0:
                os.remove(aux_sigma_path)
                print(f"Rmoeved an existing file: {aux_sigma_path}")
            else:
                with open(aux_sigma_path, "rb") as f:
                    prev_sigma = pickle.load(f)
                total_dist_sigmas.update(prev_sigma)

        with open(aux_sigma_path, "wb") as f:
            pickle.dump(total_dist_sigmas, f, -1)
        log.info(
            f"inference aux {aux_idx+1}/{aux_num} is saved to {aux_sigma_path}: element num = {len(total_dist_sigmas.keys())}")

        if os.path.exists(aux_log_weight_path):
            if aux_idx == 0:
                os.remove(aux_log_weight_path)
                print(f"Rmoeved an existing file: {aux_log_weight_path}")
            else:
                with open(aux_log_weight_path, "rb") as f:
                    prev_log_weight = pickle.load(f)
                total_log_weights.update(prev_log_weight)

        with open(aux_log_weight_path, "wb") as f:
            pickle.dump(total_log_weights, f, -1)
        log.info(
            f"inference aux {aux_idx+1}/{aux_num} is saved to {aux_sigma_path}: element num = {len(total_log_weights.keys())}")


    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_tpp.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
