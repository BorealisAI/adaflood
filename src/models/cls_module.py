# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import MulticlassCalibrationError
from lightning.pytorch.utilities import grad_norm

from src import constants
from src.utils.metrics import MeanMetricWithCount


class CLSLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimtest)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = criterion

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=net.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=net.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=net.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.l1_gradient = MeanMetricWithCount()
        self.val_loss_with_acc_best = MinMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # for calibration
        self.val_ece = MulticlassCalibrationError(
            num_classes=net.num_classes, n_bins=net.num_classes, norm='l1')
        self.test_ece = MulticlassCalibrationError(
            num_classes=net.num_classes, n_bins=net.num_classes, norm='l1')

        self.val_mce = MulticlassCalibrationError(
            num_classes=net.num_classes, n_bins=net.num_classes, norm='max')
        self.test_mce = MulticlassCalibrationError(
            num_classes=net.num_classes, n_bins=net.num_classes, norm='max')

        self.val_rmsce = MulticlassCalibrationError(
            num_classes=net.num_classes, n_bins=net.num_classes, norm='l2')
        self.test_rmsce = MulticlassCalibrationError(
            num_classes=net.num_classes, n_bins=net.num_classes, norm='l2')

        self.dataset = None
        self.tuning = False

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_loss_with_acc_best.reset()
        self.l1_gradient.reset()
        self.tmp_batch_size = 0

    def model_step(self, input_dict):
        output_dict = self.net(input_dict)
        logits = output_dict[constants.LOGITS]
        loss_dict = self.criterion(output_dict, input_dict)

        output_dict.update(loss_dict)
        output_dict.update(input_dict)
        return output_dict

    def compute_accuracy(self, input_dict, output_dict):
        labels = input_dict[constants.LABELS]
        logits = output_dict[constants.LOGITS]
        preds = torch.argmax(logits, dim=-1)
        acc = torch.mean((labels == preds).float())
        return acc.item()

    def training_step(self, input_dict, batch_idx):
        output_dict = self.model_step(input_dict)
        loss = output_dict[constants.LOSS]
        logits, labels = output_dict[constants.LOGITS], output_dict[constants.LABELS]

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(logits, labels)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.tmp_batch_size = input_dict[constants.LABELS].shape[0]

        indices = input_dict[constants.INDICES]
        losses = output_dict[constants.LOSSES]
        
        return loss

    def on_train_epoch_end(self):
        pass

    def on_before_optimizer_step(self, optimizer):
        # measure L1 norm of the gradient on training sets
        norms = grad_norm(self.net, norm_type=1)
        l1_norm = norms['grad_1.0_norm_total']
        self.l1_gradient(l1_norm, self.tmp_batch_size)
        self.log("train/l1_gradient", self.l1_gradient, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step(self, input_dict, batch_idx):
        output_dict = self.model_step(input_dict)
        loss = output_dict[constants.LOSS]
        logits, labels = output_dict[constants.LOGITS], output_dict[constants.LABELS]

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(logits, labels)
        self.val_ece(logits, labels)
        self.val_mce(logits, labels)
        self.val_rmsce(logits, labels)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ece", self.val_ece, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mce", self.val_mce, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/rmsce", self.val_rmsce, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best.update(acc)
        acc_best = self.val_acc_best.compute() # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)

        if acc == acc_best:
            self.val_loss_with_acc_best.reset()
            self.val_loss_with_acc_best(loss)
            self.log("val/loss_with_acc_best", self.val_loss_with_acc_best.compute(), prog_bar=True, sync_dist=True)
        else:
            self.log("val/loss_with_acc_best", self.val_loss_with_acc_best.compute(), prog_bar=True, sync_dist=True)

        if hasattr(self.criterion, "alpha"):
            self.log("alpha", self.criterion.alpha, prog_bar=True, sync_dist=True)
        if hasattr(self.criterion, "beta"):
            self.log("beta", self.criterion.beta, prog_bar=True, sync_dist=True)



    def test_step(self, input_dict, batch_idx):
        output_dict = self.model_step(input_dict)
        loss = output_dict[constants.LOSS]
        logits, labels = output_dict[constants.LOGITS], output_dict[constants.LABELS]

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(logits, labels)
        self.test_ece(logits, labels)
        self.test_mce(logits, labels)
        self.test_rmsce(logits, labels)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/ece", self.test_ece, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/mce", self.test_mce, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/rmsce", self.test_rmsce, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        if self.tuning:
            train_parameters = []
            for name, params in self.net.named_parameters():
                if name.startswith('fc.'):
                    train_parameters.append(params)

            optimizer = self.hparams.optimizer(params=train_parameters)
        else:
            params = [
                {'params': self.net.parameters()},
                {'params': self.criterion.parameters(), 'lr': 0.001, 'weight_decay': 0.0}]

            optimizer = self.hparams.optimizer(params=self.parameters())
            if self.hparams.scheduler is not None:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                    },
                }

        return {"optimizer": optimizer}

    def configure_optimizers_for_tuning(self, dataset):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        train_parameters = []
        for name, params in self.net.named_parameters():
            if name.startswith('fc.'):
                train_parameters.append(params)

        optimizer = self.hparams.optimizer(params=train_parameters)
        optimizer.param_groups[0]['lr'] *= 3
        return {"optimizer": optimizer}


