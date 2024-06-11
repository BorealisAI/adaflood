from typing import Any

import torch
import numpy as np
from lightning import LightningModule
from torchmetrics import MinMetric, MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src import constants
from src.utils.metrics import MeanMetricWithCount, MaskedRMSE, MaskedAccuracy, MaskedNLL

class TPPLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler=None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net
        self.criterion = criterion
        self.is_forecast = self.net.forecast_window > 0
        self.num_samples = 1

        # for averaging nll across batches
        self.train_nll = MeanMetricWithCount()
        self.val_nll = MeanMetricWithCount()
        if self.is_forecast:
            self.test_nll = MaskedNLL(self.num_samples)
        else:
            self.test_nll = MeanMetricWithCount()

        # for averaging rmse across batches
        self.train_rmse = MaskedRMSE()
        self.val_rmse = MaskedRMSE()
        self.test_rmse = MaskedRMSE()

        # for averaging accuracy across batches
        self.train_acc = MaskedAccuracy()
        self.val_acc = MaskedAccuracy()
        self.test_acc = MaskedAccuracy()

        # for tracking best so far validation nll and rmse
        self.val_nll_best = MinMetric()
        self.val_rmse_with_nll_best = MinMetric()
        self.val_acc_with_nll_best = MinMetric()

        self.val_rmse_best = MinMetric()
        self.val_nll_with_rmse_best = MinMetric()
        self.val_acc_with_rmse_best = MinMetric()

        # for tracking best so far test nll and rmse
        self.test_nll_best = MinMetric()
        self.test_rmse_best = MinMetric()
        self.test_acc_best = MinMetric()

        # for tracking kl
        self.train_kl = MeanMetric()

        self.no_decay_layer_keywords = ['cross_attn_stack']

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_nll.reset()
        self.val_nll_best.reset()
        self.val_rmse_with_nll_best.reset()
        self.val_acc_with_nll_best.reset()

        self.val_rmse.reset()
        self.val_rmse_best.reset()
        self.val_nll_with_rmse_best.reset()
        self.val_acc_with_rmse_best.reset()

        self.val_acc.reset()

    def on_test_epoch_start(self):
        self.test_nll.reset()
        self.test_nll_best.reset()

        self.test_rmse.reset()
        self.test_rmse_best.reset()

        self.test_acc.reset()
        self.test_acc_best.reset()

    #def preprocess(self, input_dict):
    #    times = input_dict[constants.TIMES]
    #    masks = input_dict[constants.MASKS]
    #    marks = input_dict[constants.MARKS]
    #    indices = input_dict[constants.INDICES]

    #    #batch_size = histories.shape[0] # B
    #    #feat_dim = histories.shape[1] # D

    #    time_batch = []
    #    mask_batch = []
    #    mark_batch = []
    #    forecast_window = self.net.forecast_window # add zeros for forecast_window

    #    for i, index_i in enumerate(indices):
    #        histories_i = []
    #        times_i = []
    #        masks_i = []
    #        marks_i = []

    #        valid_index_i = index_i[index_i >= 0]
    #        #if not forecast:
    #        #    valid_index_i = np.random.choice(valid_index_i.detach().cpu().numpy(), size=4, replace=False)

    #        for start_index in valid_index_i: # start_idx is included in the forecast_window
    #            #if not forecast:
    #            #    history_len = self.config.data.width - forecast_window
    #            #    valid_histories = histories[i][:,start_index-history_len-1:start_index+forecast_window-1]
    #            #    histories_i.append(valid_histories)
    #            #else:
    #            #zeros = torch.zeros((feat_dim, forecast_window)) #.to(histories.device)
    #            history_len = self.net.delta * 2 - forecast_window
    #            #valid_histories = histories[i][:,start_index-history_len:start_index]
    #            #extended_histories = torch.cat((valid_histories, zeros), dim=-1) # D x width
    #            #histories_i.append(extended_histories) # num_indices x D x width

    #            new_mask_i = torch.zeros_like(masks[i][start_index-history_len:start_index+forecast_window])
    #            new_mask_i[-forecast_window:] = 1 # e.g. [0, 0, 0, 0, 1, 1, 1, 1]: we compute metrics for event only in forecast window

    #            # collect time, mask and mark for history and forecast_window indicies
    #            times_i.append(times[i][start_index-history_len:start_index+forecast_window])
    #            masks_i.append(new_mask_i)
    #            marks_i.append(marks[i][start_index-history_len:start_index+forecast_window])

    #        #history_batch.append(torch.stack(histories_i)) # B x num_indices x D x width
    #        time_batch.append(torch.stack(times_i))
    #        mask_batch.append(torch.stack(masks_i))
    #        mark_batch.append(torch.stack(marks_i))

    #    #history_batch = torch.cat(history_batch).unsqueeze(1) # (N, 1, D, T) where T = width
    #    time_batch = torch.cat(time_batch) # (N, T, 1)
    #    mask_batch = torch.cat(mask_batch) # (N, T, 1)
    #    mark_batch = torch.cat(mark_batch) # (N, T, 1)

    #
    #    batch = (time_batch, mask_batch, mark_batch)
    #    return batch


    def model_step(self, input_dict):
        output_dict = self.net(**input_dict)
        if self.is_forecast:
            return output_dict

        loss_dict = self.criterion(output_dict, input_dict)
        output_dict.update(loss_dict)
        return output_dict

    def training_step(self, input_dict, batch_idx):
        output_dict = self.model_step(input_dict)

        # update nll
        loss, nlls = output_dict[constants.LOSS], output_dict[constants.NLLS]
        count = input_dict[constants.MASKS].sum()

        self.train_nll(torch.sum(nlls), count)
        self.log("train/nll", self.train_nll, on_step=False, on_epoch=True, prog_bar=True)

        kl = output_dict[constants.KL]
        if kl is not None:
            self.train_kl(kl)
            self.log("train/kl", self.train_kl, on_step=False, on_epoch=True, prog_bar=True)

        times = input_dict[constants.TIMES]
        time_preds = output_dict[constants.TIME_PREDS]
        # TODO: 1) add a discriminator that takes preds and times, 2) add optimizer_g and
        # optimizer_d, 3) compute adv loss

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, input_dict, batch_idx):
        output_dict = self.model_step(input_dict)

        # update nll
        loss, nlls = output_dict[constants.LOSS], output_dict[constants.NLLS]
        masks = input_dict[constants.MASKS].bool()
        count = masks.sum()
        self.val_nll(torch.sum(nlls), count)
        self.log("val/nll", self.val_nll, on_step=False, on_epoch=True, prog_bar=True)

        # update rmse
        times = input_dict[constants.TIMES]
        time_preds = output_dict[constants.TIME_PREDS]

        start_idx = times.shape[1] - time_preds.shape[1]
        self.val_rmse(time_preds.squeeze(-1), times[:,start_idx:].squeeze(-1),
                      masks[:,start_idx:].squeeze(-1))
        self.log("val/rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)

        # update accuracy
        class_preds = output_dict[constants.CLS_PREDS]
        if class_preds is not None:
            marks = input_dict[constants.MARKS]
            self.val_acc(class_preds, marks[:,start_idx:].squeeze(-1),
                         masks[:,start_idx:].squeeze(-1))
            self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_nll = self.val_nll.compute()
        self.val_nll_best.update(val_nll)
        val_nll_best = self.val_nll_best.compute()
        self.log("val/nll_best", self.val_nll_best.compute(), sync_dist=True, prog_bar=True)

        val_rmse = self.val_rmse.compute()
        self.val_rmse_best.update(val_rmse)
        val_rmse_best = self.val_rmse_best.compute()
        self.log("val/rmse_best", self.val_rmse_best.compute(), sync_dist=True, prog_bar=True)

        val_acc = self.val_acc.compute()

        if val_nll == val_nll_best:
            self.val_rmse_with_nll_best.reset()
            self.val_rmse_with_nll_best(val_rmse)
            self.log("val/rmse_with_nll_best", self.val_rmse_with_nll_best.compute(), sync_dist=True, prog_bar=True)
            self.val_acc_with_nll_best.reset()
            self.val_acc_with_nll_best(val_acc)
            self.log("val/acc_with_nll_best", self.val_acc_with_nll_best.compute(), sync_dist=True, prog_bar=True)
        else:
            self.log("val/rmse_with_nll_best", self.val_rmse_with_nll_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/acc_with_nll_best", self.val_acc_with_nll_best.compute(), sync_dist=True, prog_bar=True)

        #if val_rmse < prev_val_rmse_best:
        if val_rmse == val_rmse_best:
            self.val_nll_with_rmse_best.reset()
            self.val_nll_with_rmse_best(val_nll)
            self.log("val/nll_with_rmse_best", self.val_nll_with_rmse_best.compute(), sync_dist=True, prog_bar=True)
            self.val_acc_with_rmse_best.reset()
            self.val_acc_with_rmse_best(val_acc)
            self.log("val/acc_with_rmse_best", self.val_acc_with_rmse_best.compute(), sync_dist=True, prog_bar=True)
        else:
            self.log("val/nll_with_rmse_best", self.val_nll_with_rmse_best.compute(), sync_dist=True, prog_bar=True)
            self.log("val/acc_with_rmse_best", self.val_acc_with_rmse_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, input_dict, batch_idx):
        output_dict = self.model_step(input_dict)
        if self.is_forecast:
            masks = output_dict['masks']

            nlls = output_dict[constants.NLL]
            self.test_nll(nlls, masks)
            self.log("test/nll", self.test_nll, on_step=True, on_epoch=True, prog_bar=True)
            #print(f'test nll: {self.test_nll.compute()}')

            times = output_dict[constants.TIMES]
            time_preds = output_dict[constants.TIME_PREDS]
            self.test_rmse(time_preds, times, masks)
            self.log("test/rmse", self.test_rmse, on_step=True, on_epoch=True, prog_bar=True)
            #print(f'test rsme: {self.test_rmse.compute()}')

            class_preds = output_dict[constants.CLS_PREDS]
            if class_preds is not None:
                marks = output_dict[constants.MARKS]
                self.test_acc(class_preds, marks, torch.ones_like(marks).bool())
                self.log("test/acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)
                #print(f'test acc: {self.test_acc.compute()}')

            ### TODO: add MAPE
            #batch_size, seq_len, _ = times.shape
            #window_size = int(seq_len / 2)

            ## compute time_min, time_max
            #forecast_start_idx = torch.logical_not(masks).sum(1).squeeze(-1) # (B,)
            #history_last_idx = forecast_start_idx - 1
            #time_min = times[np.arange(batch_size), history_last_idx] + 1e-6

            ##times_reshaped = times.reshape(batch_size, window_size)
            ##time_preds_reshaped = time_preds.reshape(batch_size, window_size)

            #time_max = torch.min(times_reshaped[:,-1], time_preds_reshaped[:,-1]).unsqueeze(-1) + 1e-6



            #gt_num = torch.logical_and(times_reshaped >= time_min, times_reshaped < time_max).sum(-1) # (B, T) -> (B,)
            #pred_num = torch.logical_and(times_reshaped >= time_min, times_reshaped < time_max).sum(-1) # (B, T) -> (B,)

            #mape = torch.mean((gt_num - pred_num) / gt_num)
            #print(mape)


        else:
            # update nll
            loss, nlls = output_dict[constants.LOSS], output_dict[constants.NLLS]
            masks = input_dict[constants.MASKS].bool()
            count = masks.sum()
            self.test_nll.update(torch.sum(nlls), count)
            self.log("test/nll", self.test_nll, on_step=False, on_epoch=True, prog_bar=True)

            # update rmse
            times = input_dict[constants.TIMES]
            time_preds = output_dict[constants.TIME_PREDS]

            start_idx = times.shape[1] - time_preds.shape[1]
            self.test_rmse.update(
                time_preds.squeeze(-1), times[:,start_idx:].squeeze(-1), masks[:,start_idx:].squeeze(-1))
            self.log("test/rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=True)

            # update accuracy
            class_preds = output_dict[constants.CLS_PREDS]
            if class_preds is not None:
                marks = input_dict[constants.MARKS]
                self.test_acc.update(class_preds, marks[:,start_idx:].squeeze(-1),
                              masks[:,start_idx:].squeeze(-1))
                self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        # exclude pre-defined layers from weight_decay
        wd_parameters = []
        no_wd_parameters = []
        for name, params in self.net.named_parameters():
            for exclude_layer_keyword in self.no_decay_layer_keywords:
                if exclude_layer_keyword in name:
                    no_wd_parameters.append(params)
                    break
            else:
                wd_parameters.append(params)

        weight_decay = self.hparams.optimizer.keywords['weight_decay']
        params = [
            {'params': wd_parameters, 'weight_decay': weight_decay},
            {'params': no_wd_parameters, 'weight_decay': 0.0}]

        optimizer = self.hparams.optimizer(params)

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        #def configure_optimizers(self):
        #lr = self.hparams.lr
        #b1 = self.hparams.b1
        #b2 = self.hparams.b2

        #opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        #opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        #return [opt_g, opt_d], []
        return {"optimizer": optimizer}

