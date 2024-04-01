from typing import Any

import os
import time
import torch
import logging
import tensorflow as tf
import tensorboard
import numpy as np
from lightning import LightningModule
from torchmetrics import MinMetric, MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from ml_collections.config_dict import config_dict
from lightning.pytorch.core.optimizer import LightningOptimizer

from src import utils
from src import constants
from src.utils.metrics import MeanMetricWithCount, MaskedRMSE, MaskedAccuracy
from src.utils.utils import extract_n_samples

# Keep the import below for registering all model definitions
from src.models.diffuser.models import ddpm, ncsnv2 #, ncsnpp
from src.models.diffuser import losses, sampling, sde_lib, datasets
from src.models.diffuser.sampling import get_predictor, get_corrector
from src.models.diffuser.models import utils as mutils
from src.models.diffuser.models.ema import ExponentialMovingAverage
from src.models.diffuser.utils import save_checkpoint, restore_checkpoint
from src.models.diffuser.controllable_generation import get_pc_inpainter

#log = utils.get_pylogger(__name__)


class DiffusionLitModule(LightningModule):
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
        config: config_dict.ConfigDict,
        output_dir: str,
        ar_net: torch.nn.Module
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.automatic_optimization = False
        self.save_hyperparameters(logger=False, ignore=['config', 'ar_net'])
        self.config = config
        self.ar_net = ar_net
        self.ar_net.eval()

        # adjust the input dimension aligning with latent variable
        input_height = self.ar_net.embedding.weight.shape[-1]
        self.config.data.height = input_height
        self.config.data.width = self.ar_net.delta * 2

        # Create directories for experimental logs
        self.sample_dir = os.path.join(self.hparams.output_dir, "samples")
        tf.io.gfile.makedirs(self.sample_dir)

        #self.tb_dir = os.path.join(self.hparams.output_dir, "tensorboard")
        #tf.io.gfile.makedirs(self.tb_dir)
        #writer = tensorboard.SummaryWriter(self.tb_dir)

        # Initialize model.
        self.score_model = mutils.create_model(self.config)
        self.ema = ExponentialMovingAverage(
            self.score_model.parameters(), decay=self.config.model.ema_rate)
        #self.optimizer = losses.get_optimizer(self.hparams.config, score_model.parameters())
        self.state = dict(model=self.score_model, ema=self.ema, step=0)

        # Create checkpoints directory
        checkpoint_dir = os.path.join(self.hparams.output_dir, "checkpoints")

        # Intermediate checkpoints to resume training after pre-emption in cloud environments
        #checkpoint_meta_dir = os.path.join(
        #    self.hparams.output_dir, "checkpoints-meta", "checkpoint.pth")
        #tf.io.gfile.makedirs(checkpoint_dir)
        #tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))

        # Resume training when intermediate checkpoints are detected
        #self.state = restore_checkpoint(
        #    checkpoint_meta_dir, self.state, self.config.device)
        initial_step = int(self.state['step'])

        # Create data normalizer and its inverse
        self.scaler = datasets.get_data_scaler(self.config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(self.config)

        # Setup SDEs
        if self.config.training.sde.lower() == 'vpsde':
            self.sde = sde_lib.VPSDE(beta_min=self.config.model.beta_min,
                                beta_max=self.config.model.beta_max,
                                N=self.config.model.num_scales)
            sampling_eps = 1e-3
        elif self.config.training.sde.lower() == 'subvpsde':
            self.sde = sde_lib.subVPSDE(beta_min=self.config.model.beta_min,
                                   beta_max=self.config.model.beta_max,
                                   N=self.config.model.num_scales)
            sampling_eps = 1e-3
        elif self.config.training.sde.lower() == 'vesde':
            self.sde = sde_lib.VESDE(sigma_min=self.config.model.sigma_min,
                                sigma_max=self.config.model.sigma_max,
                                N=self.config.model.num_scales)
            sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {self.config.training.sde} unknown.")

        # Build one-step training and evaluation functions
        optimize_fn = losses.optimization_manager(self.config)
        continuous = self.config.training.continuous
        reduce_mean = self.config.training.reduce_mean
        likelihood_weighting = self.config.training.likelihood_weighting
        self.train_step_fn = losses.get_step_fn(
            self.sde, train=True, optimize_fn=optimize_fn, reduce_mean=reduce_mean,
            continuous=continuous, likelihood_weighting=likelihood_weighting)
        self.eval_step_fn = losses.get_step_fn(
            self.sde, train=False, optimize_fn=optimize_fn, reduce_mean=reduce_mean,
            continuous=continuous, likelihood_weighting=likelihood_weighting)

        # Building sampling functions
        sampling_shape = (self.config.training.batch_size,
                          self.config.data.num_channels,
                          self.config.data.height,
                          self.config.data.width)
        self.sampling_fn = sampling.get_sampling_fn(
            self.config, self.sde, sampling_shape, self.inverse_scaler, sampling_eps)

        num_train_steps = self.config.training.n_iters

        #log.info("Starting training loop at step %d." % (initial_step,))

        # for keeping track of diffusion losses
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for averaging nll across batches
        self.test_nll = MeanMetricWithCount()

        # for averaging rmse across batches
        self.test_rmse = MaskedRMSE()

        # for averaging accuracy across batches
        self.test_acc = MaskedAccuracy()

        # for tracking best so far test nll and rmse
        self.test_nll_best = MinMetric()
        self.test_rmse_best = MinMetric()
        self.test_acc_best = MinMetric()


        self.val_gen_min = MeanMetric()
        self.val_gen_max = MeanMetric()


    def on_train_start(self):
        self.val_loss.reset()
        self.val_gen_min.reset()
        self.val_gen_max.reset()

    def on_test_epoch_start(self):
        self.test_loss.reset()

        self.test_nll.reset()
        self.test_nll_best.reset()

        self.test_rmse.reset()
        self.test_rmse_best.reset()

        self.test_acc.reset()
        self.test_acc_best.reset()

        # to save predictions and corresponding gts
        self.time_preds = []
        self.time_gts = []

        self.class_preds = []
        self.mark_gts = []


    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict']['ema'] = self.ema

    def on_load_checkpoint(self, checkpoint):
        self.ema = checkpoint['state_dict']['ema']
        checkpoint['state_dict'].pop('ema')

        self.state = dict(model=self.score_model, ema=self.ema, step=0)
        logging.info('EMA is successfully loaded and removed from checkpoint')


    def inference_preprocess(self, histories, input_dict):
        times = input_dict[constants.TIMES]
        masks = input_dict[constants.MASKS]
        marks = input_dict[constants.MARKS]
        indices = input_dict[constants.INDICES]

        batch_size = histories.shape[0] # B
        feat_dim = histories.shape[1] # D

        time_batch = []
        mask_batch = []
        mark_batch = []
        history_batch = []
        forecast_window = self.ar_net.forecast_window # add zeros for forecast_window

        for i, index_i in enumerate(indices):
            histories_i = []
            times_i = []
            masks_i = []
            marks_i = []

            valid_index_i = index_i[index_i >= 0]

            for start_index in valid_index_i: # start_idx is included in the forecast_window
                zeros = torch.zeros((feat_dim, forecast_window)).to(histories.device)
                history_len = self.config.data.width - forecast_window
                valid_histories = histories[i][:,start_index-history_len-1:start_index-1]
                extended_histories = torch.cat((valid_histories, zeros), dim=-1) # D x width
                histories_i.append(extended_histories) # num_indices x D x width

                new_mask_i = torch.zeros_like(masks[i][start_index-history_len:start_index+forecast_window])
                new_mask_i[-forecast_window:] = 1 # e.g. [0, 0, 0, 0, 1, 1, 1, 1]: we compute metrics for event only in forecast window

                # collect time, mask and mark for history and forecast_window indicies
                times_i.append(times[i][start_index-history_len:start_index+forecast_window])
                masks_i.append(new_mask_i)
                marks_i.append(marks[i][start_index-history_len:start_index+forecast_window])

            history_batch.append(torch.stack(histories_i)) # B x num_indices x D x width
            time_batch.append(torch.stack(times_i))
            mask_batch.append(torch.stack(masks_i))
            mark_batch.append(torch.stack(marks_i))

        history_batch = torch.cat(history_batch) # (N, D, T)
        time_batch = torch.cat(time_batch) # (N, T, 1)
        mask_batch = torch.cat(mask_batch) # (N, T, 1)
        mark_batch = torch.cat(mark_batch) # (N, T, 1)
        assert history_batch.shape[1] == feat_dim
        assert history_batch.shape[2] == self.config.data.width
        assert time_batch.shape[0] == history_batch.shape[0]
        assert time_batch.shape[1] == history_batch.shape[-1]

        batch = (history_batch, time_batch, mask_batch, mark_batch)
        return batch


    def train_preprocess(self, histories, masks):
        batch_size = histories.shape[0] # B
        feat_dim = histories.shape[1] # D

        diff_window_size = self.config.data.width
        histories = histories.unsqueeze(1)

        # 1. compute max index for each sequence
        # 2. randomly select valid index to match to D
        # 3. create a new batch
        history_batch = []
        for i in range(batch_size):
            mask = masks[i]
            num_event = mask.sum().item()

            # 0-th histories is for 1-st pred -> t-th is for t+1-th pred
            # -> valid: 0 ~ num_event-1 since num_event-1-th history is for num_event-th pred
            if num_event-1-diff_window_size <= 0:
                #log.warn(f'num_event - 1 < diff_window_size')
                continue

            start_idx = int(np.random.choice(
                np.arange(0, num_event-1-diff_window_size, 1), size=1).item())

            history_batch.append(histories[i,:,:,start_idx:start_idx+diff_window_size])

        history_batch = torch.stack(history_batch, dim=0)
        return history_batch

    #def on_after_backward(self) -> None:
    #    print("on_after_backward enter")
    #    for name, p in self.named_parameters():
    #        if p.grad is not None:
    #            print(name)
    #    print("on_after_backward exit")

    def encode(self, input_dict, forecast=False):
        with torch.no_grad():
            output_dict = self.ar_net.encode(**input_dict)
        histories = output_dict[constants.HISTORIES].transpose(1, 2) # B x T x D -> B x D x T

        if constants.INDICES in input_dict:
            indices = input_dict[constants.INDICES]
        else:
            indices = []

        if forecast and len(indices) > 0:
            batch = self.inference_preprocess(histories, input_dict)
        else:
            batch = self.train_preprocess(histories, input_dict[constants.MASKS])
        return batch

    def generate_samples(self, input_dict, clip=True): # This will be shared between val and test. For val, we have self.ema from training but for test, we need to load a checkpoint for ema check line 276 - 285 in run_lib.py
        self.ema.store(self.score_model.parameters()) # store score_model to ema backup
        self.ema.copy_to(self.score_model.parameters()) # copy ema main params to score_model

        predictor = get_predictor(self.config.sampling.predictor.lower())
        corrector = get_corrector(self.config.sampling.corrector.lower())

        pc_inpainter = get_pc_inpainter(
            self.sde, predictor, corrector, inverse_scaler=self.inverse_scaler, snr=self.config.sampling.snr,
            n_steps=1, probability_flow=False, continuous=self.config.training.continuous,
            denoise=True, eps=1e-5, num_resample=1)

        forecast_window = self.ar_net.forecast_window # add zeros for forecast_window
        # Collect a batch using starting indices
        (history_batch, time_batch, mask_batch, mark_batch) = self.encode(
            input_dict, forecast=True)

        mini_batch_size = self.config.eval.batch_size
        num_sampling_rounds = history_batch.shape[0] // mini_batch_size + 1

        # Generate samples on forecast windows
        inpainted_batch = []
        start_time = time.time()
        for r in range(num_sampling_rounds):
            if r > 0: break
            #import IPython; IPython.embed(); exit()
            print(f'inpainting {r+1}/{num_sampling_rounds}')
            batch_i = history_batch[r*mini_batch_size:(r+1)*mini_batch_size].unsqueeze(1) # (mini_batch_size, 1, D, T)
            mask = torch.ones_like(batch_i)
            mask[:, :, :, -forecast_window:] = 0.
            inpainted_batch_i = pc_inpainter(
                self.score_model, self.scaler(batch_i), mask)
            print(f'Min: {inpainted_batch_i.min()}, Max: {inpainted_batch_i.max()}')
            if clip:
                inpainted_batch_i = torch.clamp(inpainted_batch_i, min=-1.0, max=1.0)
            inpainted_batch.append(inpainted_batch_i.squeeze(1))

        end_time = time.time()
        print(f'inpainting took {end_time - start_time}')

        inpainted_batch = torch.cat(inpainted_batch).transpose(1, 2) # (B, D, T) -> (B, T, D)
        self.ema.restore(self.score_model.parameters()) # restore ema backup params to score_model

        #import IPython; IPython.embed()
        # DEBUG: for debugging purpose
        time_batch = time_batch[:inpainted_batch.shape[0]]
        mask_batch = mask_batch[:inpainted_batch.shape[0]]
        mark_batch = mark_batch[:inpainted_batch.shape[0]]

        batch = (inpainted_batch, time_batch, mask_batch, mark_batch)
        return batch


        #mask = torch.ones_like(batch)
        #mask_width = int(self.config.data.width / 2)
        #mask[:, :, :, mask_width:] = 0.


            #this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
            #tf.io.gfile.makedirs(this_sample_dir)
            #nrow = int(np.sqrt(sample.shape[0]))
            #image_grid = make_grid(sample, nrow, padding=2)

            # TODO: check sample shape if we need to permute or not
            #sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)

        #if config.eval.enable_sampling:
        #  for r in range(num_sampling_rounds):
        #    logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        #    # Directory to save samples. Different for each host to avoid writing conflicts
        #    this_sample_dir = os.path.join(
        #      eval_dir, f"ckpt_{ckpt}")
        #    tf.io.gfile.makedirs(this_sample_dir)
        #    samples, n = sampling_fn(score_model)
        #    samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        #    samples = samples.reshape(
        #      (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        #    # Write samples to disk or Google Cloud Storage
        #    with tf.io.gfile.GFile(
        #        os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
        #      io_buffer = io.BytesIO()
        #      np.savez_compressed(io_buffer, samples=samples)
        #      fout.write(io_buffer.getvalue())

            ## Force garbage collection before calling TensorFlow code for Inception network
            #gc.collect()
            #latents = evaluation.run_inception_distributed(samples, inception_model,
            #                                               inceptionv3=inceptionv3)
            ## Force garbage collection again before returning to JAX code
            #gc.collect()
            ## Save latent represents of the Inception network to disk or Google Cloud Storage
            #with tf.io.gfile.GFile(
            #    os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
            #  io_buffer = io.BytesIO()
            #  np.savez_compressed(
            #    io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
            #  fout.write(io_buffer.getvalue())

          # Compute inception scores, FIDs and KIDs.
          # Load all statistics that have been previously computed and saved for each host
          #all_logits = []
          #all_pools = []
          #this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
          #stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
          #for stat_file in stats:
          #  with tf.io.gfile.GFile(stat_file, "rb") as fin:
          #    stat = np.load(fin)
          #    if not inceptionv3:
          #      all_logits.append(stat["logits"])
          #    all_pools.append(stat["pool_3"])

          #if not inceptionv3:
          #  all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
          #all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

          ## Load pre-computed dataset statistics.
          #data_stats = evaluation.load_dataset_stats(config)
          #data_pools = data_stats["pool_3"]

          ## Compute FID/KID/IS on all samples together.
          #if not inceptionv3:
          #  inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
          #else:
          #  inception_score = -1

          #fid = tfgan.eval.frechet_classifier_distance_from_activations(
          #  data_pools, all_pools)
          ## Hack to get tfgan KID work for eager execution.
          #tf_data_pools = tf.convert_to_tensor(data_pools)
          #tf_all_pools = tf.convert_to_tensor(all_pools)
          #kid = tfgan.eval.kernel_classifier_distance_from_activations(
          #  tf_data_pools, tf_all_pools).numpy()
          #del tf_data_pools, tf_all_pools

          #logging.info(
          #  "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
          #    ckpt, inception_score, fid, kid))

          #with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
          #                       "wb") as f:
          #  io_buffer = io.BytesIO()
          #  np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
          #  f.write(io_buffer.getvalue())



    def training_step(self, input_dict, batch_idx):
        batch = self.encode(input_dict)
        loss = self.train_step_fn(self.state, batch, self)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        pass


    def validation_step(self, input_dict, batch_idx):
        batch = self.encode(input_dict, forecast=False)
        loss = self.eval_step_fn(self.state, batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # DEBUG:
        #if batch_idx == 0:
        #    one_sample_input_dict = extract_n_samples(input_dict, n=1)
        #    (inpainted_encoding_batch, time_batch, mask_batch, mark_batch) =\
        #        self.generate_samples(one_sample_input_dict, clip=False)
        #    gen_min, gen_max = inpainted_encoding_batch.min(), inpainted_encoding_batch.max()
        #    self.val_gen_min(gen_min)
        #    self.val_gen_max(gen_max)
        #    self.log("val/gen_min", self.val_gen_min, on_step=False, on_epoch=True, prog_bar=True)
        #    self.log("val/gen_max", self.val_gen_max, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, input_dict, batch_idx):
        batch = self.encode(input_dict, forecast=False)
        loss = self.eval_step_fn(self.state, batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        print(f'test loss: {loss}')

        (inpainted_encoding_batch, time_batch, mask_batch, mark_batch) = self.generate_samples(
            input_dict, clip=True)
        encode_out = {constants.HISTORIES: inpainted_encoding_batch}

        output_dict = self.ar_net(
            times=time_batch, marks=mark_batch, masks=mask_batch, encode_out=encode_out)

        nlls = output_dict[constants.NLL]
        nll, count = nlls.sum(), nlls.numel()
        self.test_nll.update(nll, count)
        self.log("test/nll", self.test_nll, on_step=True, on_epoch=True, prog_bar=True)
        print(f'event size: {nlls.shape[0]}')
        print(f'test nll: {self.test_nll.compute()}')

        times = output_dict[constants.TIMES]
        time_preds = output_dict[constants.TIME_PREDS]
        self.test_rmse.update(
            time_preds, times, torch.ones_like(times).bool())
        self.log("test/rmse", self.test_rmse, on_step=True, on_epoch=True, prog_bar=True)
        print(f'test rsme: {self.test_rmse.compute()}')

        class_preds = output_dict[constants.CLS_PREDS]
        if class_preds is not None:
            marks = output_dict[constants.MARKS]
            self.test_acc.update(class_preds, marks,torch.ones_like(marks).bool())
            self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
            print(f'test acc: {self.test_acc.compute()}')

        # Keep track of times, time_preds, class_preds and marks
        self.time_preds.append(time_preds.detach().cpu().numpy())
        self.time_gts.append(times.detach().cpu().numpy())

        if class_preds is not None:
            self.class_preds.append(class_preds.detach().cpu().numpy())
            self.mark_gts.append(marks.detach().cpu().numpy())


    def on_test_epoch_end(self):
        # Save times, time_preds, class_preds and marks
        np.save(os.path.join(self.sample_dir, 'time_gts.npy'), self.time_gts)
        np.save(os.path.join(self.sample_dir, 'time_preds.npy'), self.time_preds)
        if self.class_preds:
            np.save(os.path.join(self.sample_dir, 'class_preds.npy'), self.class_preds)
            np.save(os.path.join(self.sample_dir, 'class_gts.npy'), self.marks_gts)


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = losses.get_optimizer(
            self.config, self.score_model.parameters())
        self.state['optimizer'] = LightningOptimizer(optimizer)

        return {"optimizer": optimizer}

