""" This module defines some network classes for selective capacity models. """
import os
import logging
import copy
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from lightning import LightningModule
import torch.distributed as dist

from src import constants
from src.models.tpp import util
from src.models.tpp.prob_dists import NormalMixture, LogNormalMixture
from src.models.tpp.flow import ContinuousGRULayer, ContinuousLSTMLayer
from src.models.tpp.thp.models import (
    TransformerEncoder, TransformerAttnEncoder, NPVIEncoder, NPMLEncoder,
    TransformerRNN, TransformerDecoder)
from src.models.tpp.thp import util as thp_util

logger = logging.getLogger(__name__)


class IntensityFreePredictor(LightningModule):
    def __init__(self, name, dataset_name, d_model, num_components, num_classes, flow=None,
                 activation=None, weights_path=None, perm_invar=False, compute_acc=True,
                 train_diffuser=False, forecast_window=1):
        '''
        d_model: the size of intermediate features
        num_components: the number of mixtures
        encoder: dictionary that specifices arguments for the encoder
        activation: dictionary that specifices arguments for the activation function
        '''
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.compute_acc = compute_acc
        self.delta = constants.DELTAS[dataset_name]
        self.forecast_window = forecast_window

        self.perm_invar = perm_invar
        self.d_model = d_model
        self.embedding = nn.Embedding(self.num_classes+1, d_model, padding_idx=constants.PAD)

        # if flow is specified, it correponds to neural flow else intensity-free
        self.flow = flow
        if self.flow == 'gru':
            self.encoder = ContinuousGRULayer(
                1 + d_model, hidden_dim=d_model,
                model='flow', flow_model='resnet', flow_layers=1,
                hidden_layers=2, time_net='TimeTanh', time_hidden_dime=8)
        elif self.flow == 'lstm':
            self.encoder = ContinuousLSTMLayer(
                1 + d_model, hidden_dim=d_model+1,
                model='flow', flow_model='resnet', flow_layers=1,
                hidden_layers=2, time_net='TimeTanh', time_hidden_dime=8)
        else:
            self.encoder = nn.GRU(
                1 + d_model, d_model, batch_first=True)
        self.activation = util.build_activation(activation)

        if self.perm_invar:
            decoder_hidden_dim = self.d_model * 2
        else:
            decoder_hidden_dim = self.d_model

        self.prob_dist = LogNormalMixture(
            decoder_hidden_dim, num_components, activation=self.activation)

        if self.num_classes > 1:
            self.mark_linear = nn.Linear(decoder_hidden_dim, self.num_classes)

        self.train_diffuser = train_diffuser

        if weights_path is not None:
            checkpoint = torch.load(weights_path)['state_dict']
            try:
                self.load_state_dict(checkpoint)
            except:
                checkpoint = {key.replace("net.", "", 1): value for key, value in checkpoint.items() if key.startswith("net.")}
                self.load_state_dict(checkpoint)


    def encode(self, times, marks, masks, missing_masks=[], indices=[], with_last=False):
        marks_emb = self.embedding(marks.squeeze(-1)) # (B, Seq, D)
        inputs = torch.cat([times, marks_emb], -1) # , Seq, D+1)

        # obtain the features from the encoder
        if self.flow != 'gru' and self.flow != 'lstm':
            hidden = torch.zeros(
                1, 1, self.d_model).repeat(1, times.shape[0], 1).to(times.device) # (1, B, D)
            histories, _ = self.encoder(inputs, hidden) # (B, Seq, D)
        else:
            histories = self.encoder(inputs, times)

        if not with_last:
            histories = histories[:,:-1] # (B, Seq-1, D)

        encode_out = {
            constants.HISTORIES: histories}
        return encode_out

    def diffuse(self, encode_out):
        return

    def decode(self, times, masks, marks, encode_out, forecast=False, last_only=False):
        histories = encode_out[constants.HISTORIES]
        if not forecast: # forecast: False -> for regular tpp
            prob_output_dict = self.prob_dist(
                histories, times[:,1:], masks[:,1:]) # (B, Seq-1, 1): ignore the first event since that's only input not output
        elif last_only: # forecast: True, last_only: True - > for event-by-event tpp
            prob_output_dict = self.prob_dist.forecast(
                histories, times, masks, last_only=True)
        else: # forecast: True, last_only: False -> for multi-step diff
            prob_output_dict = self.prob_dist.forecast(
                histories, times, masks, last_only=False) # (B, Seq, 1)

        event_ll = prob_output_dict['event_ll']
        surv_ll = prob_output_dict['surv_ll']
        time_predictions = prob_output_dict['preds']
        log_weights = prob_output_dict['log_weights']
        lognorm_dist = prob_output_dict['lognorm_dist']
        dist_mu = prob_output_dict['mu']
        dist_sigma = prob_output_dict['sigma']

        # compute log-likelihood and class predictions if marks are available
        class_log_probs = None
        class_logits = None
        class_predictions = None
        if self.num_classes > 1 and self.compute_acc:
            batch_size = times.shape[0]
            last_event_idx = masks.squeeze(-1).sum(-1, keepdim=True).long().squeeze(-1) - 1 # (batch_size,)

            class_logits = torch.log_softmax(self.mark_linear(histories), dim=-1)  # (B, Seq-1 or Seq, num_marks)
            mark_dist = Categorical(logits=class_logits)
            adjusted_marks = torch.where(marks-1 >= 0, marks-1, torch.zeros_like(marks)).squeeze(-1) # original dataset uses 1-index
            if forecast:
                class_log_probs = mark_dist.log_prob(adjusted_marks)  # (B, Seq)
                class_predictions = torch.argmax(class_logits, dim=-1)

                class_log_probs = class_log_probs.unsqueeze(-1)[masks.bool()]
                class_predictions = class_predictions.unsqueeze(-1)[masks.bool()]
            else:
                masks_without_last = masks.clone()
                masks_without_last[torch.arange(batch_size), last_event_idx] = 0

                class_log_probs = mark_dist.log_prob(adjusted_marks[:,1:])  # (B, Seq-1)
                class_log_probs = torch.stack(
                    [torch.mean(mark_log_prob[mask.bool()]) for
                     mark_log_prob, mask in zip(class_log_probs, masks_without_last.squeeze(-1)[:,:-1])]) # (B,)

                class_predictions = torch.argmax(class_logits, dim=-1)

        output_dict = {
            constants.EVENT_LL: event_ll,
            constants.SURV_LL: surv_ll,
            constants.KL: None,
            constants.TIME_PREDS: time_predictions,
            constants.CLS_LL: class_log_probs,
            constants.CLS_LOGITS: class_logits,
            constants.CLS_PREDS: class_predictions,
            constants.LOG_WEIGHTS: log_weights,
            constants.LOGNORM_DIST: lognorm_dist,
            constants.DIST_MU: dist_mu,
            constants.DIST_SIGMA: dist_sigma,
        }
        return output_dict

    def forecast(self, times, marks, masks, missing_masks=[], indices=[]):
        batch_size = times.shape[0]

        start_time = time.time()
        total_time_preds_in_window = []
        total_times_in_window = []
        total_mark_preds_in_window = []
        total_marks_in_window = []
        total_nll_in_window = []

        for batch_idx in range(batch_size):
            #print(f'Processing: {batch_idx}/{batch_size} batch')
            time_i = times[batch_idx].unsqueeze(0) # 1 x T x 1
            mark_i = marks[batch_idx].unsqueeze(0) # 1 x T x 1
            mask_i = masks[batch_idx].unsqueeze(0) # 1 x T x 1
            if missing_masks:
                missing_mask_i = missing_masks[batch_idx].unsqueeze(0) # 1 x T x 1
            else:
                missing_mask_i = missing_masks
            index_i = indices[batch_idx]
            seq_len = mask_i.sum().item()

            # TODO: make it batch computation
            for k, start_idx in enumerate(index_i):
                nll_in_window = []
                if start_idx < 0: break
                subset_time_i = time_i[:,:start_idx].clone() # start_idx is included in the forecast_window
                subset_mark_i = mark_i[:,:start_idx].clone()
                subset_mask_i = mask_i[:,:start_idx].clone()
                if missing_masks:
                    subset_missing_mask_i = missing_mask_i[:,:start_idx].clone()
                else:
                    subset_missing_mask_i = missing_masks

                forecast_window = min(self.forecast_window, seq_len - start_idx)

                for j in range(forecast_window):
                    current_idx = start_idx + j
                    subset_encode_out = self.encode(
                        subset_time_i, subset_mark_i, subset_mask_i,
                        subset_missing_mask_i, indices=[], with_last=True)
                    subset_decode_out = self.decode(
                        times=subset_time_i, marks=subset_mark_i, masks=subset_mask_i,
                        encode_out=subset_encode_out, forecast=True, last_only=True)

                    time_pred = subset_decode_out[constants.TIME_PREDS]
                    #time_pred = time_preds.squeeze()[-1]

                    mark_preds = subset_decode_out[constants.CLS_PREDS]
                    if mark_preds is not None:
                        mark_pred = mark_preds.squeeze()[-1]
                    else:
                        mark_pred = None

                    event_ll = subset_decode_out[constants.EVENT_LL]
                    nll = -event_ll
                    mark_ll = subset_decode_out[constants.CLS_LL]
                    if self.num_classes > 1 and mark_ll is not None:
                        nll -= mark_ll.squeeze()[-1]
                    nll_in_window.append(nll)

                    # update subsets
                    subset_time_i = torch.cat(
                        (subset_time_i, time_pred.view(1, -1, 1)), dim=1)
                    if mark_pred is not None:
                        subset_mark_i = torch.cat(
                            (subset_mark_i, mark_pred.view(1, -1, 1) + 1), dim=1) # gt marks are 1-indexed
                    else:
                        subset_mark_i = mark_i[:,:current_idx+1].clone() # dummy

                    subset_mask_i = mask_i[:,:current_idx+1].clone() # dummy
                    if missing_masks: # dummy
                        subset_missing_mask_i = missing_mask_i[:,:start_idx+1].clone()
                    else:
                        subset_missing_mask_i = missing_masks

                    assert subset_time_i.shape[1] == current_idx + 1

                assert subset_time_i.shape[1] == start_idx + forecast_window
                # collect predictions and corresponding times and marks
                time_preds_in_window = subset_time_i[
                    :,-forecast_window:].reshape(forecast_window)
                times_in_window = time_i[
                    :,start_idx:start_idx+forecast_window].reshape(forecast_window)

                mark_preds_in_window = subset_mark_i[
                    :,-forecast_window:].reshape(forecast_window) - 1 # pred marks are 0-indexed
                marks_in_window = mark_i[
                    :,start_idx:start_idx+forecast_window].reshape(forecast_window)

                total_time_preds_in_window.append(time_preds_in_window)
                total_times_in_window.append(times_in_window)

                total_mark_preds_in_window.append(mark_preds_in_window)
                total_marks_in_window.append(marks_in_window)

                total_nll_in_window.append(torch.tensor(nll_in_window).to(times.device))

        total_time_preds_in_window = torch.cat(total_time_preds_in_window, dim=0)
        total_times_in_window = torch.cat(total_times_in_window, dim=0)

        if self.num_classes > 1:
            total_mark_preds_in_window = torch.cat(total_mark_preds_in_window, dim=0)
            total_marks_in_window = torch.cat(total_marks_in_window, dim=0)
        else:
            total_mark_preds_in_window = None
            total_marks_in_window = None

        total_nll_in_window = torch.cat(total_nll_in_window, dim=0)
        #print(f'It took {time.time() - start_time} sec')

        forecast_out = {
            constants.TIME_PREDS: total_time_preds_in_window,
            constants.TIMES: total_times_in_window,
            constants.CLS_PREDS: total_mark_preds_in_window,
            constants.MARKS: total_marks_in_window,
            constants.NLL: total_nll_in_window
        }
        return forecast_out


    def forecast_multi(self, times, marks, masks, missing_masks=[], indices=[], encode_out=None):
        output_dict = self.decode(
            times=times, masks=masks, marks=marks, encode_out=encode_out, forecast=True, last_only=False)

        valid_time_preds = output_dict[constants.TIME_PREDS]
        valid_times = times[masks.bool()]

        valid_class_predictions = output_dict[constants.CLS_PREDS]
        valid_marks = marks[masks.bool()]

        nll = -output_dict[constants.EVENT_LL]
        cls_ll = output_dict[constants.CLS_LL]
        if self.num_classes > 1 and cls_ll is not None:
            nll -= cls_ll

        forecast_out = {
            constants.TIME_PREDS: valid_time_preds,
            constants.TIMES: valid_times,
            constants.CLS_PREDS: valid_class_predictions,
            constants.MARKS: valid_marks,
            constants.NLL: nll
        }
        return forecast_out


    def forward(self, times, marks, masks, missing_masks=[], indices=[], encode_out=None):
        if isinstance(missing_masks, torch.Tensor):
            masks = torch.logical_and(masks.bool(), missing_masks.bool()).float()

        if len(indices) > 0: # event-by-event forecast
            forecast_out = self.forecast(
                times=times, marks=marks, masks=masks, missing_masks=missing_masks,
                indices=indices)
            return forecast_out
        elif encode_out is not None: # multi-step diff
            forecast_out = self.forecast_multi(
                times=times, marks=marks, masks=masks, missing_masks=missing_masks,
                indices=indices, encode_out=encode_out)
            return forecast_out
        else:
            encode_out = self.encode(times, marks, masks, missing_masks, indices)
            decode_out = self.decode(times, masks, marks, encode_out)
            return decode_out


class TransformerMix(LightningModule):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, name, dataset_name, activation, num_classes, d_model=256, d_inner=1024,
                 n_layers=2, cattn_n_layers=1, n_head=4, d_k=64, d_v=64, dropout=0.1,
                 attn_l=0, base_l=20, perm_invar=False, use_avg=True, share_weights=True,
                 attn_only=False, concat=False, num_latent=0, vi_method=None,
                 num_z_samples=100, compute_acc=True, weights_path=None, forecast_window=0):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.compute_acc = compute_acc
        self.delta = constants.DELTAS[dataset_name]
        self.forecast_window = forecast_window

        self.encoder = TransformerEncoder(
            num_types=num_classes,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            base_l=base_l
        )

        self.num_latent = num_latent
        # npvi refers to vi and npml refers to mc approximation in meta TPP
        try:
            self.vi_method = eval(vi_method)
        except:
            self.vi_method = vi_method

        if self.vi_method is not None:
            assert num_latent > 0

        if self.num_latent > 0 and self.vi_method is not None:
            if self.vi_method == 'npvi':
                self.latent_encoder = NPVIEncoder(
                    d_model, self.num_latent, num_z_samples=num_z_samples)
            elif self.vi_method == 'npml':
                self.latent_encoder = NPMLEncoder(
                    d_model, self.num_latent, num_z_samples=num_z_samples)
            else:
                logger.error(f'VI method - {self.vi_method} is not valid')

        self.perm_invar = perm_invar
        self.use_avg = use_avg
        self.attn_only = attn_only
        self.attn_encoder = None
        if attn_l > 0:
            self.attn_encoder = TransformerAttnEncoder(
                num_types=num_classes,
                d_model=d_model,
                d_inner=d_inner,
                n_layers=n_layers,
                n_head=n_head,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
                attn_l=attn_l,
                cattn_n_layers=cattn_n_layers,
                concat=concat
            )

            if not self.attn_only and concat:
                d_model = int(d_model * 1.5)
            elif concat:
                d_model = int(d_model * 2)
            elif self.perm_invar:
                d_model = int(d_model * 1.5)

            if not self.attn_only:
                d_model = int(d_model * 2)

            if share_weights:
                self.attn_encoder.layer_stack = self.encoder.layer_stack
        else:
            if self.perm_invar:
                d_model = int(d_model * 2)

        self.num_classes = num_classes

        if self.num_classes > 1:
            self.mark_linear = nn.Linear(d_model, self.num_classes)

        self.activation = util.build_activation(activation)
        self.prob_dist = LogNormalMixture(
            d_model, components=8, activation=self.activation, vi_method=self.vi_method)

        trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The number of trainable model parameters: {trainable_params}', flush=True)

        if weights_path is not None:
            checkpoint = torch.load(weights_path)['state_dict']
            try:
                self.load_state_dict(checkpoint)
            except:
                checkpoint = {key.replace("net.", "", 1): value for key, value in checkpoint.items() if key.startswith("net.")}
                self.load_state_dict(checkpoint)

    def encode(self, times, marks, masks, missing_masks=[], indices=[]):
        batch_size = times.shape[0]
        # obtain the features from the transformer encoder
        encode_out, _ = self.encoder(times, marks, masks)

        # compute the global feature G
        if self.perm_invar:
            target_inputs = encode_out
            zeros = torch.zeros((encode_out.shape[0], 1, encode_out.shape[2])).to(times.device)
            context_encode = torch.cat((zeros, encode_out[:,:-1]), dim=1)
            context_encode = torch.cumsum(context_encode, dim=1)
            if self.use_avg:
                num_cum_seq = torch.arange(1, context_encode.shape[1]+1).reshape(1, -1, 1).to(times.device)
                encode_out = context_encode / num_cum_seq
            else:
                encode_out = context_encode

        # sample latent variable z and compute kl if it is npvi
        kl = None
        latent_out = None
        if self.num_latent > 0 and self.vi_method is not None:
            mus, vars, encode_out = self.latent_encoder(encode_out)

            if self.vi_method == 'npvi' and self.training:
                last_event_idx = masks.squeeze(-1).sum(-1, keepdim=True).long().squeeze(-1) - 1 # (batch_size,)
                encode_out = encode_out[:,torch.arange(batch_size), last_event_idx].unsqueeze(2)
                encode_out = encode_out.repeat(1, 1, times.shape[-1], 1)
                prior_mu = mus[:,:-1]
                prior_log_sigma = vars[:,:-1]
                last_event_idx = masks.squeeze(-1).sum(-1, keepdim=True).long().squeeze(-1) - 1
                posterior_mu = mus[torch.arange(batch_size), last_event_idx].unsqueeze(1).repeat(
                    1, prior_mu.shape[1], 1)
                posterior_log_sigma = vars[torch.arange(batch_size), last_event_idx].unsqueeze(1).repeat(
                    1, prior_log_sigma.shape[1], 1)
                kl = self.kl_div(
                    prior_mu, prior_log_sigma, posterior_mu, posterior_log_sigma, masks)

        # obtain the features from the attention encoder
        if self.attn_encoder:
            attn_encode_out, attentions = self.attn_encoder(times, marks, masks)
            if self.attn_only:
                encode_out = attn_encode_out
            else:
                if self.vi_method is not None:
                    attn_encode_out = attn_encode_out.unsqueeze(0)
                    attn_encode_out = attn_encode_out.expand_as(encode_out)
                encode_out = torch.cat([encode_out, attn_encode_out], dim=-1)

        # THP+ baseline does not take target inputs into account. It's applicable only to meta TPP
        if self.perm_invar:
            if self.vi_method is not None:
                target_inputs = target_inputs.repeat(encode_out.shape[0], 1, 1, 1)
            encode_out = torch.cat([encode_out, target_inputs], dim=-1)

        if self.vi_method is not None:
            histories = encode_out[:,:,:-1] # (L, B, Seq-1, D): ignore the last time as it is xmax and L: num of z samples
        else:
            histories = encode_out[:,:-1] # (B, Seq-1, D): ignore the last time as it is xmax and L: num of z samples

        encode_out = {
            constants.HISTORIES: histories,
            constants.KL: kl}
        return encode_out

    def decode(self, times, masks, marks, encode_out):
        histories = encode_out[constants.HISTORIES]
        # obatin log-likelihood from the mixture of log-normals
        prob_output_dict = self.prob_dist(
            histories, times[:,1:], masks[:,1:]) # (B, Seq-1, 1): ignore the first event since that's only input not output

        event_ll = prob_output_dict['event_ll']
        surv_ll = prob_output_dict['surv_ll']
        time_predictions = prob_output_dict['preds']
        log_weights = prob_output_dict['log_weights']
        lognorm_dist = prob_output_dict['lognorm_dist']
        dist_mu = prob_output_dict['mu']
        dist_sigma = prob_output_dict['sigma']

        # compute log-likelihood and class predictions if marks are available
        class_log_probs = None
        class_logits = None
        class_predictions = None
        if self.num_classes > 1 and self.compute_acc:
            batch_size = times.shape[0]
            last_event_idx = masks.squeeze(-1).sum(-1, keepdim=True).long().squeeze(-1) - 1 # (batch_size,)
            masks_without_last = masks.clone()
            masks_without_last[torch.arange(batch_size), last_event_idx] = 0

            class_logits = torch.log_softmax(self.mark_linear(histories), dim=-1)  # (B, Seq-1, num_marks)
            class_dist = Categorical(logits=class_logits)
            adjusted_marks = torch.where(marks-1 >= 0, marks-1, torch.zeros_like(marks)) # original dataset uses 1-index
            class_log_probs = class_dist.log_prob(adjusted_marks[:,1:])  # (B, Seq-1)

            #probs = torch.softmax(class_logits, dim=-1)
            #flattened_probs, flattened_marks = (
            #    probs.reshape(-1, self.num_classes), adjusted_marks[:,1:].reshape(-1))
            #true_class_probs = flattened_probs[torch.arange(flattened_marks.shape[0]), flattened_marks]
            #true_class_probs = true_class_probs.reshape(batch_size, -1)
            #true_class_log_probs = torch.log(true_class_probs)

            # if it is a latent variable model, class predictions are mode of prediction samples
            if self.vi_method is not None:
                masks_without_last = masks_without_last.squeeze(-1)[:,:-1]
                num_z_samples = class_log_probs.shape[0]
                class_log_probs = torch.stack(
                    [torch.sum(torch.logsumexp(class_log_probs[:,i,masks_without_last[i].bool()], dim=0) - torch.log(torch.tensor(num_z_samples)))
                     for i in range(batch_size)])
                class_predictions = torch.mode(torch.argmax(class_logits, dim=-1), dim=0)[0]
            else:
                class_log_probs = torch.stack(
                    [torch.sum(mark_log_prob[mask.bool()]) for
                     mark_log_prob, mask in zip(class_log_probs, masks_without_last.squeeze(-1)[:,:-1])])
                class_predictions = torch.argmax(class_logits, dim=-1)

        output_dict = {
            constants.HISTORIES: histories,
            constants.EVENT_LL: event_ll,
            constants.SURV_LL: surv_ll,
            constants.KL: encode_out[constants.KL],
            constants.TIME_PREDS: time_predictions,
            constants.CLS_LL: class_log_probs,
            constants.CLS_LOGITS: class_logits,
            constants.CLS_PREDS: class_predictions,
            constants.LOG_WEIGHTS: log_weights,
            constants.LOGNORM_DIST: lognorm_dist,
            constants.DIST_MU: dist_mu,
            constants.DIST_SIGMA: dist_sigma,
        }
        return output_dict

    def forward(self, times, marks, masks, missing_masks=[], indices=[]):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_{N-1}), we predict (l_2, ..., l_N).
        Input: times: (B, Seq, 1);
               marks: (B, Seq, 1);
               masks: (B, Seq, 1).
        """
        times = times.squeeze(-1)
        marks = marks.squeeze(-1)

        encode_out = self.encode(times, marks, masks, missing_masks=missing_masks, indices=indices)
        decode_out = self.decode(times, masks, marks, encode_out)
        return decode_out


    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var, masks):
        kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = kl_div * masks[:,:-1]
        kl_div = 0.5 * kl_div.sum(dim=(1, 2))
        return kl_div


