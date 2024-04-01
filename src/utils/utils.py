import os
import glob
import warnings
from importlib.util import find_spec
from typing import Callable

from omegaconf import DictConfig

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


def redirect_base_ckpt_path(ckpt_path: str) -> str:
    splits = ckpt_path.split('/')
    base_ckpt_path = '/'.join(splits[:-3] + splits[-2:])
    return base_ckpt_path

def extract_best_ckpt_path(ckpt_path: str) -> str:
    ckpt_dir = '/'.join(ckpt_path.split('/')[:-1])
    ckpt_list = glob.glob(os.path.join(ckpt_dir, 'epoch_*.ckpt'))

    latest_version = -1
    # find the latest version
    for ckpt_path_i in ckpt_list:
        ckpt_name = ckpt_path_i.split('/')[-1].split('.')[0]
        splits = ckpt_name.split('_')
        if '-' in splits[-1]:
            epoch_num, version = splits[-1].split('-')
        else:
            epoch_num = splits[-1]
            version = None
        #if len(splits) == 2:
        #    _, epoch_num = splits
        #    version = None
        #elif len(splits) == 3:
        #    _, epoch_num, version = splits
        #else:
        #    raise NotImplementedError("Extracting best checkpoint is wrong")

        if version is not None:
            version = int(version[1:]) # remove v
            if version > latest_version:
                latest_version = version

    # filter checkpoints with the latest version
    latest_ckpt_list = []
    if latest_version == -1:
        for ckpt_path_i in ckpt_list:
            latest_ckpt_list.append(ckpt_path_i)
    else:
        for ckpt_path_i in ckpt_list:
            ckpt_name = ckpt_path_i.split('/')[-1].split('.')[0]
            splits = ckpt_name.split('_')
            if '-' not in splits[-1]:
                continue
            #if len(splits) != 3:
            #    continue
            else:
                epoch_num, version = splits[-1].split('-')
                #_, epoch_num, version = splits
                version = int(version[1:]) # remove v
                if version == latest_version:
                    latest_ckpt_list.append(ckpt_path_i)

    best_ckpt_path = None
    latest_epoch = -1
    # find the latext epoch
    for ckpt_path_i in latest_ckpt_list:
        ckpt_name = ckpt_path_i.split('/')[-1].split('.')[0]
        splits = ckpt_name.split('_')
        if '-' in splits[-1]:
            epoch_num, version = splits[-1].split('-')
        else:
            epoch_num = splits[-1]
            version = None

        epoch_num = int(epoch_num)
        if epoch_num > latest_epoch:
            latest_epoch = epoch_num
            best_ckpt_path = ckpt_path_i

    return best_ckpt_path


def extract_n_samples(input_dict, n=1):
    new_input_dict = {}
    for key in input_dict:
        new_input_dict[key] = input_dict[key][:n]
    return new_input_dict

