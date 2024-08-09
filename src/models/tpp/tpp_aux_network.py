# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle
import logging
import torch
import numpy as np

from src import constants
from src.models.tpp.tpp_network import IntensityFreePredictor, TransformerMix
from src.models.tpp.prob_dists import LogNormal

logger = logging.getLogger(__name__)


class IntensityFreePredictorWithAux(IntensityFreePredictor):
    def __init__(self, name, d_model, num_components, num_classes, flow=None,
                 activation=None, weights_path=None, perm_invar=False, compute_acc=True,
                 aux_logit_path=None, aux_lr=None, aux_weight_decay=None, aux_d_model=None):
        '''
        d_model: the size of intermediate features
        num_components: the number of mixtures
        encoder: dictionary that specifices arguments for the encoder
        activation: dictionary that specifices arguments for the activation function
        '''
        super().__init__(
            name=name, d_model=d_model, num_components=num_components,
            num_classes=num_classes, flow=flow, activation=activation,
            weights_path=weights_path, perm_invar=perm_invar, compute_acc=compute_acc)

        aux_loss_path = aux_logit_path.replace('_logits', '_losses')
        aux_mu_path = aux_logit_path.replace('_logits', '_mus')
        aux_sigma_path = aux_logit_path.replace('_logits', '_sigmas')
        aux_log_weight_path = aux_logit_path.replace('_logits', '_log_weights')

        assert os.path.exists(aux_loss_path), f"aux_logit_path: {aux_loss_path} does not exist"

        if os.path.exists(aux_logit_path):
            with open(aux_logit_path, "rb") as f:
                self.logit_dict = pickle.load(f)
        else:
            self.logit_dict = None

        with open(aux_loss_path, "rb") as f:
            self.loss_dict = pickle.load(f)

        with open(aux_mu_path, "rb") as f:
            self.mu_dict = pickle.load(f)

        with open(aux_sigma_path, "rb") as f:
            self.sigma_dict = pickle.load(f)

        with open(aux_log_weight_path, "rb") as f:
            self.log_weight_dict = pickle.load(f)

        self.aux_min_loss = np.min([loss for loss in self.loss_dict.values()])

        assert os.path.exists(aux_loss_path), f"aux_logit_path: {aux_loss_path} does not exist"


    def forward(self, times, marks, masks, missing_masks=[], indices=[]):
        aux_output_dict = {}
        if self.training:
            if self.logit_dict is not None:
                np_aux_logits = np.stack(
                    [self.logit_dict[idx.item()] for idx in indices], axis=0)
                aux_logits = torch.from_numpy(np_aux_logits).to(times.device)
                aux_output_dict = {constants.AUX_LOGITS: aux_logits}

            np_aux_losses = np.stack(
                [self.loss_dict[idx.item()] for idx in indices], axis=0)
            aux_losses = torch.from_numpy(np_aux_losses).to(times.device)
            aux_output_dict.update({constants.AUX_LOSSES: aux_losses})

            np_aux_mus = np.stack(
                [self.mu_dict[idx.item()] for idx in indices], axis=0)
            aux_mus = torch.from_numpy(np_aux_mus).to(times.device)

            np_aux_sigmas = np.stack(
                [self.sigma_dict[idx.item()] for idx in indices], axis=0)
            aux_sigmas = torch.from_numpy(np_aux_sigmas).to(times.device)

            aux_prob_dist = LogNormal(aux_mus, aux_sigmas)
            aux_output_dict.update({constants.AUX_PROB_DIST: aux_prob_dist})

            aux_output_dict.update({constants.PROB_DIST: self.prob_dist})

            np_aux_log_weights = np.stack(
                [self.log_weight_dict[idx.item()] for idx in indices], axis=0)
            aux_log_weights = torch.from_numpy(np_aux_log_weights).to(times.device)

            aux_output_dict.update({constants.AUX_LOG_WEIGHTS: aux_log_weights})

            aux_output_dict.update({'aux_min_loss': self.aux_min_loss})

        output_dict = super().forward(
            times=times, marks=marks, masks=masks, missing_masks=missing_masks)
        output_dict.update(aux_output_dict)
        return output_dict


class TransformerMixWithAux(TransformerMix):
    def __init__(self, name, activation, num_classes, d_model=256, d_inner=1024,
                 n_layers=2, cattn_n_layers=1, n_head=4, d_k=64, d_v=64, dropout=0.1,
                 attn_l=0, base_l=20, perm_invar=False, use_avg=True, share_weights=True,
                 attn_only=False, concat=False, num_latent=0, vi_method=None, num_z_samples=100,
                 compute_acc=True, aux_logit_path=None, aux_lr=None, aux_weight_decay=None, aux_d_model=None):

        '''
        hidden_dim: the size of intermediate features
        num_components: the number of mixtures
        encoder: dictionary that specifices arguments for the encoder
        activation: dictionary that specifices arguments for the activation function
        '''
        super().__init__(
            name=name, activation=activation, num_classes=num_classes, d_model=d_model,
            d_inner=d_inner, n_layers=n_layers, cattn_n_layers=cattn_n_layers, n_head=n_head,
            d_k=d_k, d_v=d_v, dropout=dropout, attn_l=attn_l, base_l=base_l, perm_invar=perm_invar,
            use_avg=use_avg, share_weights=share_weights, attn_only=attn_only, concat=concat,
            num_latent=num_latent, vi_method=vi_method, num_z_samples=num_z_samples, compute_acc=compute_acc)

        aux_loss_path = aux_logit_path.replace('_logits', '_losses')
        #aux_pred_path = aux_logit_path.replace('_logits', '_preds')
        aux_mu_path = aux_logit_path.replace('_logits', '_mus')
        aux_sigma_path = aux_logit_path.replace('_logits', '_sigmas')
        aux_log_weight_path = aux_logit_path.replace('_logits', '_log_weights')

        assert os.path.exists(aux_loss_path), f"aux_logit_path: {aux_loss_path} does not exist"

        if os.path.exists(aux_logit_path):
            with open(aux_logit_path, "rb") as f:
                self.logit_dict = pickle.load(f)
        else:
            self.logit_dict = None

        with open(aux_loss_path, "rb") as f:
            self.loss_dict = pickle.load(f)

        #with open(aux_pred_path, "rb") as f:
        #    self.pred_dict = pickle.load(f)

        with open(aux_mu_path, "rb") as f:
            self.mu_dict = pickle.load(f)

        with open(aux_sigma_path, "rb") as f:
            self.sigma_dict = pickle.load(f)

        with open(aux_log_weight_path, "rb") as f:
            self.log_weight_dict = pickle.load(f)

        self.aux_min_loss = np.min([loss for loss in self.loss_dict.values()])

    def forward(self, times, marks, masks, missing_masks=[], indices=[]):
        aux_output_dict = {}
        if self.training:
            if self.logit_dict is not None:
                np_aux_logits = np.stack(
                    [self.logit_dict[idx.item()] for idx in indices], axis=0)
                aux_logits = torch.from_numpy(np_aux_logits).to(times.device)
                aux_output_dict = {constants.AUX_LOGITS: aux_logits}

            np_aux_losses = np.stack(
                [self.loss_dict[idx.item()] for idx in indices], axis=0)
            aux_losses = torch.from_numpy(np_aux_losses).to(times.device)
            aux_output_dict.update({constants.AUX_LOSSES: aux_losses})

            np_aux_mus = np.stack(
                [self.mu_dict[idx.item()] for idx in indices], axis=0)
            aux_mus = torch.from_numpy(np_aux_mus).to(times.device)

            np_aux_sigmas = np.stack(
                [self.sigma_dict[idx.item()] for idx in indices], axis=0)
            aux_sigmas = torch.from_numpy(np_aux_sigmas).to(times.device)

            aux_prob_dist = LogNormal(aux_mus, aux_sigmas)
            aux_output_dict.update({constants.AUX_PROB_DIST: aux_prob_dist})

            aux_output_dict.update({constants.PROB_DIST: self.prob_dist})

            np_aux_log_weights = np.stack(
                [self.log_weight_dict[idx.item()] for idx in indices], axis=0)
            aux_log_weights = torch.from_numpy(np_aux_log_weights).to(times.device)

            aux_output_dict.update({constants.AUX_LOG_WEIGHTS: aux_log_weights})

            aux_output_dict.update({'aux_min_loss': self.aux_min_loss})

        output_dict = super().forward(
            times=times, marks=marks, masks=masks, missing_masks=missing_masks)
        output_dict.update(aux_output_dict)
        return output_dict

