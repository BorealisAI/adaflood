# MIT License

# Copyright (c) 2021 ashleve

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import glob
import torch
import warnings
from importlib.util import find_spec
from typing import Callable
import numpy as np

from omegaconf import DictConfig

from src import constants
from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

def load_checkpoint_path(checkpoint_dir):
    # list files ends with .pth
    # if there is only one that starts with epoch, load it if not load the last
    # if no checkpoints start with epoch, load last.ckpt
    checkpoint_list = glob.glob(f'{checkpoint_dir}/*.ckpt')
    assert checkpoint_list

    latest_epoch = 0
    selected_checkpoint_path = os.path.join(checkpoint_dir, 'last.ckpt')
    for checkpoint_path in checkpoint_list:
        if checkpoint_path.split('/')[-1].startswith('epoch'):
            curr_epoch = int(checkpoint_path.split('_')[-1][:3])
            if curr_epoch > latest_epoch:
                latest_epoch = curr_epoch
                selected_checkpoint_path = checkpoint_path

    log.info(f"Loading a checkopint from {selected_checkpoint_path}")
    return selected_checkpoint_path

def collate_fn(batch):
    input_dict = {
        constants.IMAGES: torch.stack([x[0] for x in batch]).float(),
        constants.LABELS: torch.tensor([x[1] for x in batch]).long()
    }
    return input_dict

def to_device(tensor_dict, device='cuda'):
    for key in tensor_dict.keys():
        if isinstance(tensor_dict[key], torch.Tensor):
            tensor_dict[key] = tensor_dict[key].to(device)
    return tensor_dict

def generate_noisy_labels(labels, num_classes):
    noisy_labels = []
    for label in labels:
        cand_labels = np.array(list(
            set(np.arange(num_classes).tolist()).difference(set([label]))))
        noisy_label = np.random.choice(cand_labels, size=1, replace=False)
        noisy_labels.append(noisy_label)

    noisy_labels = np.concatenate(noisy_labels)
    return noisy_labels

def generate_noisy_labels_subgroup(labels, num_classes, alpha):
    label_maps = {0: [8], 1: [9], 2: [], 3: [5], 4: [7], 5: [3], 6: [], 7: [4], 8: [0], 9: [1]}
    alphas = {0: alpha + 0.3, 1: alpha + 0.2, 2: 0.0, 3: alpha + 0.1, 4: alpha,
              5: alpha + 0.1, 6: 0.0, 7: alpha, 8: alpha + 0.3, 9: alpha + 0.2}

    noisy_labels = []
    for label in labels:
        cand_labels = label_maps[int(label)]
        if cand_labels:
            flip_p = alphas[int(label)]
            is_flip = np.random.binomial(n=1, p=flip_p, size=1).astype(bool).item()
            if is_flip:
                noisy_label = np.random.choice(cand_labels, size=1, replace=False).item()
            else:
                noisy_label = label.item()
        else:
            noisy_label = label.item()

        noisy_labels.append(noisy_label)

    noisy_labels = np.stack(noisy_labels)
    return noisy_labels

def find_latest_version(ckpt_path):
    ckpt_dir = '/'.join(ckpt_path.split('/')[:-1])
    ckpt_paths = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))

    latest_ckpt_path = None
    latest_version = 0
    for path in ckpt_paths:
        file_name = path.split('/')[-1]
        if not file_name.startswith('last'):
            continue

        version = file_name.split('-')[-1]
        if version == 'last':
            continue

        try:
            version = int(version.split('.')[0][1:])
        except:
            continue
        if version > latest_version:
            latest_version = version
            latest_ckpt_path = path

    if latest_ckpt_path is None:
        latest_ckpt_path = ckpt_path
    return latest_ckpt_path






