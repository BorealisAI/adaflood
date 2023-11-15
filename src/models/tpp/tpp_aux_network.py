""" This module defines some network classes for selective capacity models. """
import os
import pickle
import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from lightning import LightningModule

from src import constants
from src.models.tpp import util
from src.models.tpp.prob_dists import NormalMixture, LogNormalMixture
from src.models.tpp.flow import ContinuousGRULayer, ContinuousLSTMLayer
from src.models.tpp.thp.models import (
    TransformerEncoder, TransformerAttnEncoder, NPVIEncoder, NPMLEncoder,
    TransformerRNN, TransformerDecoder)
from src.models.tpp.thp import util as thp_util
from src.models.tpp.tpp_network import IntensityFreePredictor, TransformerMix
from src.utils.utils import load_checkpoint_path
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

            #np_aux_preds = np.stack(
            #    [self.pred_dict[idx.item()] for idx in indices], axis=0)
            #aux_preds = torch.from_numpy(np_aux_preds).to(times.device)
            #aux_output_dict.update({constants.AUX_PREDS: aux_preds})

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

            #np_aux_losses = np.stack(
            #    [self.loss_dict[idx.item()] for idx in indices], axis=0)
            #aux_losses = torch.from_numpy(np_aux_losses).to(times.device)
            #aux_output_dict = {constants.AUX_LOSSES: aux_losses}
            #import IPython; IPython.embed()

        output_dict = super().forward(
            times=times, marks=marks, masks=masks, missing_masks=missing_masks)
        output_dict.update(aux_output_dict)
        return output_dict





    #    aux1_weights_path = load_checkpoint_path(aux1_weights_dir)
    #    aux2_weights_path = load_checkpoint_path(aux2_weights_dir)

    #    self.aux_model1 = IntensityFreePredictor(
    #        name=name, hidden_dim=aux_d_model, num_components=num_components,
    #        num_classes=num_classes, flow=flow, activation=activation,
    #        weights_path=aux1_weights_path, perm_invar=perm_invar, compute_acc=compute_acc)

    #    self.aux_model2 = IntensityFreePredictor(
    #        name=name, hidden_dim=aux_d_model, num_components=num_components,
    #        num_classes=num_classes, flow=flow, activation=activation,
    #        weights_path=aux2_weights_path, perm_invar=perm_invar, compute_acc=compute_acc)

    #def forward(self, times, marks, masks, missing_masks=[], is_first_half=[]):
    #    with torch.inference_mode():
    #        aux1_output_dict = self.aux_model1.forward(times, marks, masks, missing_masks, is_first_half)
    #        aux2_output_dict = self.aux_model2.forward(times, marks, masks, missing_masks, is_first_half)
    #        aux1_output_dict = {
    #            'aux1_' + key: val for key, val in aux1_output_dict.items()}
    #        aux2_output_dict = {
    #            'aux2_' + key: val for key, val in aux2_output_dict.items()}

    #        aux1_output_dict.update(aux2_output_dict)

    #    output_dict = super().forward(times, marks, masks, missing_masks, is_first_half)
    #    output_dict.update(aux1_output_dict)
    #    return output_dict


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

            #np_aux_preds = np.stack(
            #    [self.pred_dict[idx.item()] for idx in indices], axis=0)
            #aux_preds = torch.from_numpy(np_aux_preds).to(times.device)
            #aux_output_dict.update({constants.AUX_PREDS: aux_preds})

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

            #np_aux_losses = np.stack(
            #    [self.loss_dict[idx.item()] for idx in indices], axis=0)
            #aux_losses = torch.from_numpy(np_aux_losses).to(times.device)
            #aux_output_dict = {constants.AUX_LOSSES: aux_losses}
            #import IPython; IPython.embed()

        output_dict = super().forward(
            times=times, marks=marks, masks=masks, missing_masks=missing_masks)
        output_dict.update(aux_output_dict)
        return output_dict



        #with torch.inference_mode():
        #    aux_output_dict = self.aux_model.forward(times, marks, masks, missing_masks, is_first_half)
        #    aux_output_dict = {
        #        'aux_' + key: val for key, val in aux_output_dict.items()}

        #output_dict = super().forward(times, marks, masks, missing_masks, is_first_half)
        #output_dict.update(aux_output_dict)
        #return output_dict



    #def forward(self, times, marks, masks, missing_masks=[], is_first_half=[]):
    #    with torch.inference_mode():
    #        aux1_output_dict = self.aux_model1.forward(times, marks, masks, missing_masks, is_first_half)
    #        aux2_output_dict = self.aux_model2.forward(times, marks, masks, missing_masks, is_first_half)
    #        aux1_output_dict = {
    #            'aux1_' + key: val for key, val in aux1_output_dict.items()}
    #        aux2_output_dict = {
    #            'aux2_' + key: val for key, val in aux2_output_dict.items()}

    #        aux1_output_dict.update(aux2_output_dict)

    #    output_dict = super().forward(times, marks, masks, missing_masks, is_first_half)
    #    output_dict.update(aux1_output_dict)
    #    return output_dict


