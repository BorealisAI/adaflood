# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple

import time
import shutil
import copy
import os
import hydra
import pickle
import numpy as np
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
from src.models.cls.resnet_aux import ResBaseAux

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
    log.info(f"PID {os.getpid()}")

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

    ckpt_path = cfg.get("ckpt_path")

    if cfg.get("train") and not ckpt_path:
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    # for aux, save prediction results
    # (load dataset with exclusion indices, make predictions on them and save them)
    aux_idx, aux_num = cfg['data']['aux_idx'], cfg['data']['aux_num']

    metric_dict = {}
    if cfg.get("test"):
        log.info("Starting testing!")
        #ckpt_path = trainer.checkpoint_callback.best_model_path
        if not ckpt_path:
            ckpt_path = trainer.checkpoint_callback.best_model_path

        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    # merge train and test metrics

    if aux_num == 0 and not isinstance(model.net, ResBaseAux):
        model.to('cuda')
        model.eval()
        aux_train_dataloader = datamodule.train_dataloader()

        # set paths for losses and logits
        aux_output_dir = os.path.abspath(
            os.path.join(cfg["paths"]["output_dir"], os.pardir))
        aux_train_loss_path = os.path.join(
            aux_output_dir, f"aux{aux_num}_eval_losses.pkl") # pretend it's eval
        aux_train_logit_path = os.path.join(
            aux_output_dir, f"aux{aux_num}_eval_logits.pkl") # pretend it's eval

        total_train_losses = {}
        total_train_logits = {}
        total_train_corrects = []
        for i, input_dict in enumerate(aux_train_dataloader):
            with torch.no_grad():
                output_dict = model.model_step(utils.to_device(input_dict))

                # save indices, losses and logits
                train_indices = input_dict[constants.INDICES]
                train_losses = output_dict[constants.LOSSES]
                train_logits = output_dict[constants.LOGITS]

                train_logit_dict = {
                    idx.item(): logit.detach().cpu().numpy() for idx, logit in zip(
                        train_indices, train_logits)}
                train_loss_dict = {
                    idx.item(): loss.detach().cpu().numpy() for idx, loss in zip(
                        train_indices, train_losses)}

                total_train_losses.update(train_loss_dict)
                total_train_logits.update(train_logit_dict)

                # compute correctness for sanity check
                train_labels = input_dict[constants.LABELS]
                train_corrects = torch.eq(torch.argmax(train_logits, dim=1), train_labels)
                total_train_corrects.append(train_corrects)

                if i % 100 == 0:
                    print(f'{i}/{len(aux_train_dataloader)} processed')

        # sanity check for performance
        total_train_corrects = torch.cat(total_train_corrects)
        print(f'Aux Train ACC: {torch.mean(total_train_corrects.float())}')

        # save losses
        replace_dict = {}
        if os.path.exists(aux_train_loss_path):
            if aux_idx == 0:
                os.remove(aux_train_loss_path)
                print(f"Rmoeved an existing file: {aux_train_loss_path}")
            else:
                with open(aux_train_loss_path, "rb") as f:
                    prev_losses = pickle.load(f)

                # save a larger train loss
                overlap_indices = np.intersect1d(
                    np.array(list(prev_losses.keys())), np.array(list(total_train_losses.keys())))

                for overlap_idx in overlap_indices:
                    prev_loss = prev_losses[overlap_idx]
                    curr_loss = total_train_losses[overlap_idx]
                    if curr_loss > prev_loss:
                        replace_dict[overlap_idx] = True
                        prev_losses[overlap_idx] = curr_loss
                    else:
                        replace_dict[overlap_idx] = False
                total_train_losses.update(prev_losses)

        with open(aux_train_loss_path, "wb") as f:
            pickle.dump(total_train_losses, f, pickle.HIGHEST_PROTOCOL)
        log.info(
            f"Aux Train {aux_idx+1}/{aux_num} is saved to \
              {aux_train_loss_path}: element num = {len(total_train_losses.keys())}")

        # save logits
        if os.path.exists(aux_train_logit_path):
            if aux_idx == 0:
                os.remove(aux_train_logit_path)
                print(f"Removed an existing file: {aux_train_logit_path}")
            else:
                with open(aux_train_logit_path, "rb") as f:
                    prev_logits = pickle.load(f)

                if replace_dict:
                    for overlap_idx in replace_dict.keys():
                        if replace_dict[overlap_idx]:
                            prev_logits[overlap_idx] = total_train_logits[overlap_idx]
                total_train_logits.update(prev_logits)

        with open(aux_train_logit_path, "wb") as f:
            pickle.dump(total_train_logits, f, pickle.HIGHEST_PROTOCOL)
        log.info(
            f"Aux Train {aux_idx+1}/{aux_num} is saved to \
              {aux_train_logit_path}: element num = {len(total_train_logits.keys())}")


    if aux_num == -2 and not isinstance(model.net, ResBaseAux):
        trainer_checkpoint_callback_best_model_path = trainer.checkpoint_callback.best_model_path
        del trainer
        print('trainer deleted')
        if not trainer_checkpoint_callback_best_model_path and not ckpt_path:
            print(f'{trainer_checkpoint_callback_best_model_path} path does not exist'); exit()
        if ckpt_path:
            trainer_checkpoint_callback_best_model_path = ckpt_path

        total_start_time = time.time()

        dataset = datamodule.dataset
        # set paths for losses and logits
        aux_output_dir = os.path.abspath(
            os.path.join(cfg["paths"]["output_dir"], os.pardir))
        aux_eval_loss_path = os.path.join(
            aux_output_dir, f"aux{aux_num}_eval_losses.pkl")
        aux_eval_logit_path = os.path.join(
            aux_output_dir, f"aux{aux_num}_eval_logits.pkl")

        tuning_epochs = 25
        train_dataset = datamodule.data_train
        num_train_dataset = len(train_dataset)
        batch_size = int(num_train_dataset / 10.0)
        shuffle_indices = np.arange(num_train_dataset)
        np.random.shuffle(shuffle_indices)
        val_dataloader = datamodule.val_dataloader()

        def _train_indices(indices, num_dataset):
            return np.setdiff1d(np.arange(num_dataset), indices)

        tmp_trainer_cfg = copy.deepcopy(cfg.trainer)
        tmp_trainer_cfg['max_epochs'] = tuning_epochs
        datamodule.batch_size = 128

        tmp_callback_cfg = copy.deepcopy(cfg.callbacks)
        tmp_callback_cfg.model_checkpoint.dirpath = os.path.join(
            cfg.callbacks.model_checkpoint.dirpath, 'tmp')

        total_eval_losses = {}
        total_eval_logits = {}
        for k in range(0, num_train_dataset, batch_size):
            if os.path.exists(tmp_callback_cfg.model_checkpoint.dirpath):
                shutil.rmtree(tmp_callback_cfg.model_checkpoint.dirpath, ignore_errors=True)
            os.makedirs(tmp_callback_cfg.model_checkpoint.dirpath, exist_ok=True)

            log.info(f"Instantiating model <{cfg.model._target_}>")
            model: LightningModule = hydra.utils.instantiate(cfg.model)
            model.tuning = True
            model.dataset = dataset

            log.info("Instantiating callbacks...")
            tmp_callbacks: List[Callback] = utils.instantiate_callbacks(tmp_callback_cfg)

            tmp_trainer: Trainer = hydra.utils.instantiate(tmp_trainer_cfg, callbacks=tmp_callbacks, logger=logger)

            start_time = time.time()
            eval_indices = shuffle_indices[k:k+batch_size]
            train_indices = _train_indices(eval_indices, num_train_dataset)
            tmp_train_dataloader = datamodule.indexed_dataloader(train_indices)

            best_ckpt = torch.load(trainer_checkpoint_callback_best_model_path)['state_dict']
            model.load_state_dict(best_ckpt)
            del best_ckpt

            # reset params 
            model.net.fc.reset_parameters()

            model.train()
            tmp_trainer.fit(model=model, train_dataloaders=tmp_train_dataloader, val_dataloaders=val_dataloader)
            torch.cuda.empty_cache()

            model.to('cpu')
            tmp_best_ckpt = torch.load(tmp_trainer.checkpoint_callback.best_model_path)['state_dict']
            for key, v in tmp_best_ckpt.items():
                tmp_best_ckpt[key] = v.cpu()

            model.load_state_dict(tmp_best_ckpt)
            del tmp_best_ckpt

            print(f"[{k}/{num_train_dataset}] Loaded a checkpoint from {tmp_trainer.checkpoint_callback.best_model_path}")

            model.to('cuda')
            model.eval()
            tmp_eval_dataloader = datamodule.indexed_dataloader(eval_indices)
            for input_dict in tmp_eval_dataloader:
                output_dict = model.model_step(utils.to_device(input_dict))
                losses = output_dict[constants.LOSSES]
                logits = output_dict[constants.LOGITS]

                total_eval_losses.update(
                    {idx.item(): loss.detach().cpu().numpy() for idx, loss in zip(
                        input_dict[constants.INDICES], losses)})
                total_eval_logits.update(
                    {idx.item(): logit.detach().cpu().numpy() for idx, logit in zip(
                        input_dict[constants.INDICES], logits)})

                acc = model.compute_accuracy(input_dict, output_dict)
                print(f'Eval accuracy: {acc}')

            print(f'[{k}/{num_train_dataset}] completed in {time.time() - start_time}')

        # save losses
        if os.path.exists(aux_eval_loss_path):
            os.remove(aux_eval_loss_path)
            print(f"Rmoeved an existing file: {aux_eval_loss_path}")

        with open(aux_eval_loss_path, "wb") as f:
            pickle.dump(total_eval_losses, f, pickle.HIGHEST_PROTOCOL)
        log.info(
            f"Aux Eval is saved to \
              {aux_eval_loss_path}: element num = {len(total_eval_losses.keys())}")

        # save logits
        if os.path.exists(aux_eval_logit_path):
            os.remove(aux_eval_logit_path)
            print(f"Rmoeved an existing file: {aux_eval_logit_path}")

        with open(aux_eval_logit_path, "wb") as f:
            pickle.dump(total_eval_logits, f, pickle.HIGHEST_PROTOCOL)
        log.info(
            f"Aux Eval is saved to \
              {aux_eval_logit_path}: element num = {len(total_eval_logits.keys())}")

        total_end_time = time.time()
        print(f'It took {total_end_time - total_start_time}')


    if aux_idx >= 0:
        total_start_time = time.time()

        model.to('cuda')
        model.eval()
        aux_eval_dataloader = datamodule.axu_infer_dataloader()
        aux_train_dataloader = datamodule.train_dataloader()

        # set paths for losses and logits
        aux_output_dir = os.path.abspath(
            os.path.join(cfg["paths"]["output_dir"], os.pardir))
        aux_eval_loss_path = os.path.join(
            aux_output_dir, f"aux{aux_num}_eval_losses.pkl")
        aux_eval_logit_path = os.path.join(
            aux_output_dir, f"aux{aux_num}_eval_logits.pkl")
        aux_train_loss_path = os.path.join(
            aux_output_dir, f"aux{aux_num}_train_losses.pkl")
        aux_train_logit_path = os.path.join(
            aux_output_dir, f"aux{aux_num}_train_logits.pkl")

        # compute aux losses for unseen data
        total_eval_losses = {}
        total_eval_logits = {}
        total_eval_corrects = []

        for i, input_dict in enumerate(aux_eval_dataloader):
            with torch.no_grad():
                output_dict = model.model_step(utils.to_device(input_dict))

                # save indices, losses and logits
                eval_indices = input_dict[constants.INDICES]
                eval_losses = output_dict[constants.LOSSES]
                eval_logits = output_dict[constants.LOGITS]

                eval_logit_dict = {
                    idx.item(): logit.detach().cpu().numpy() for idx, logit in zip(
                        eval_indices, eval_logits)}
                eval_loss_dict = {
                    idx.item(): loss.detach().cpu().numpy() for idx, loss in zip(
                        eval_indices, eval_losses)}

                total_eval_losses.update(eval_loss_dict)
                total_eval_logits.update(eval_logit_dict)

                # compute correctness for sanity check
                eval_labels = input_dict[constants.LABELS]
                eval_corrects = torch.eq(torch.argmax(eval_logits, dim=1), eval_labels)
                total_eval_corrects.append(eval_corrects)

        # sanity check for performance
        total_eval_corrects = torch.cat(total_eval_corrects)
        print(f'Aux Eval ACC: {torch.mean(total_eval_corrects.float())}')

        # save losses
        if os.path.exists(aux_eval_loss_path):
            if aux_idx == 0:
                os.remove(aux_eval_loss_path)
                print(f"Rmoeved an existing file: {aux_eval_loss_path}")
            else:
                with open(aux_eval_loss_path, "rb") as f:
                    prev_losses = pickle.load(f)
                total_eval_losses.update(prev_losses)

        with open(aux_eval_loss_path, "wb") as f:
            pickle.dump(total_eval_losses, f, pickle.HIGHEST_PROTOCOL)
        log.info(
            f"Aux Eval {aux_idx+1}/{aux_num} is saved to \
              {aux_eval_loss_path}: element num = {len(total_eval_losses.keys())}")

        # save logits
        if os.path.exists(aux_eval_logit_path):
            if aux_idx == 0:
                os.remove(aux_eval_logit_path)
                print(f"Rmoeved an existing file: {aux_eval_logit_path}")
            else:
                with open(aux_eval_logit_path, "rb") as f:
                    prev_logits = pickle.load(f)
                total_eval_logits.update(prev_logits)

        with open(aux_eval_logit_path, "wb") as f:
            pickle.dump(total_eval_logits, f, pickle.HIGHEST_PROTOCOL)
        log.info(
            f"Aux Eval {aux_idx+1}/{aux_num} is saved to \
              {aux_eval_logit_path}: element num = {len(total_eval_logits.keys())}")


        # compute aux losses for seen data
        last_ckpt = torch.load(trainer.checkpoint_callback.last_model_path)['state_dict']
        model.load_state_dict(last_ckpt)
        model.to('cuda')
        model.eval()
        print(f"Loaded a checkpoint from {trainer.checkpoint_callback.last_model_path}")

        total_train_losses = {}
        total_train_logits = {}
        total_train_corrects = []
        for i, input_dict in enumerate(aux_train_dataloader):
            with torch.no_grad():
                output_dict = model.model_step(utils.to_device(input_dict))

                # save indices, losses and logits
                train_indices = input_dict[constants.INDICES]
                train_losses = output_dict[constants.LOSSES]
                train_logits = output_dict[constants.LOGITS]

                train_logit_dict = {
                    idx.item(): logit.detach().cpu().numpy() for idx, logit in zip(
                        train_indices, train_logits)}
                train_loss_dict = {
                    idx.item(): loss.detach().cpu().numpy() for idx, loss in zip(
                        train_indices, train_losses)}

                total_train_losses.update(train_loss_dict)
                total_train_logits.update(train_logit_dict)

                # compute correctness for sanity check
                train_labels = input_dict[constants.LABELS]
                train_corrects = torch.eq(torch.argmax(train_logits, dim=1), train_labels)
                total_train_corrects.append(train_corrects)

        # sanity check for performance
        total_train_corrects = torch.cat(total_train_corrects)
        print(f'Aux Train ACC: {torch.mean(total_train_corrects.float())}')

        # save losses
        replace_dict = {}
        if os.path.exists(aux_train_loss_path):
            if aux_idx == 0:
                os.remove(aux_train_loss_path)
                print(f"Rmoeved an existing file: {aux_train_loss_path}")
            else:
                with open(aux_train_loss_path, "rb") as f:
                    prev_losses = pickle.load(f)

                # save a larger train loss
                overlap_indices = np.intersect1d(
                    np.array(list(prev_losses.keys())), np.array(list(total_train_losses.keys())))

                for overlap_idx in overlap_indices:
                    prev_loss = prev_losses[overlap_idx]
                    curr_loss = total_train_losses[overlap_idx]
                    if curr_loss > prev_loss:
                        replace_dict[overlap_idx] = True
                        prev_losses[overlap_idx] = curr_loss
                    else:
                        replace_dict[overlap_idx] = False
                total_train_losses.update(prev_losses)

        with open(aux_train_loss_path, "wb") as f:
            pickle.dump(total_train_losses, f, pickle.HIGHEST_PROTOCOL)
        log.info(
            f"Aux Train {aux_idx+1}/{aux_num} is saved to \
              {aux_train_loss_path}: element num = {len(total_train_losses.keys())}")

        # save logits
        if os.path.exists(aux_train_logit_path):
            if aux_idx == 0:
                os.remove(aux_train_logit_path)
                print(f"Rmoeved an existing file: {aux_train_logit_path}")
            else:
                with open(aux_train_logit_path, "rb") as f:
                    prev_logits = pickle.load(f)

                if replace_dict:
                    for overlap_idx in replace_dict.keys():
                        if replace_dict[overlap_idx]:
                            prev_logits[overlap_idx] = total_train_logits[overlap_idx]
                total_train_logits.update(prev_logits)

        with open(aux_train_logit_path, "wb") as f:
            pickle.dump(total_train_logits, f, pickle.HIGHEST_PROTOCOL)
        log.info(
            f"Aux Train {aux_idx+1}/{aux_num} is saved to \
              {aux_train_logit_path}: element num = {len(total_train_logits.keys())}")

        total_end_time = time.time()
        print(f'It took {total_end_time - total_start_time}')

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_cls.yaml")
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
