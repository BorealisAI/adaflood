""" This module defines some network classes for selective capacity models. """
import os
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

logger = logging.getLogger(__name__)

class IntensityFreePredictorWithAux(IntensityFreePredictor):
    def __init__(self, name, hidden_dim, num_components, num_classes, flow=None,
                 activation=None, weights_path=None, perm_invar=False, compute_acc=True,
                 aux1_weights_dir=None, aux2_weights_dir=None,
                 aux_lr=None, aux_weight_decay=None, aux_d_model=None):
                 #alpha_init=1.0, beta_init=0.0, affine_trainable=False):
        '''
        hidden_dim: the size of intermediate features
        num_components: the number of mixtures
        encoder: dictionary that specifices arguments for the encoder
        activation: dictionary that specifices arguments for the activation function
        '''
        super().__init__(
            name=name, hidden_dim=hidden_dim, num_components=num_components,
            num_classes=num_classes, flow=flow, activation=activation,
            weights_path=weights_path, perm_invar=perm_invar, compute_acc=compute_acc)

        #if affine_trainable:
        #    self.alpha = nn.Parameter(torch.tensor(alpha_init))
        #    self.beta = nn.Parameter(torch.tensor(beta_init))
        #else:
        #    self.alpha = torch.tensor(alpha_init)
        #    self.beta = torch.tensor(beta_init)

        aux1_weights_path = load_checkpoint_path(aux1_weights_dir)
        aux2_weights_path = load_checkpoint_path(aux2_weights_dir)

        self.aux_model1 = IntensityFreePredictor(
            name=name, hidden_dim=aux_d_model, num_components=num_components,
            num_classes=num_classes, flow=flow, activation=activation,
            weights_path=aux1_weights_path, perm_invar=perm_invar, compute_acc=compute_acc)

        self.aux_model2 = IntensityFreePredictor(
            name=name, hidden_dim=aux_d_model, num_components=num_components,
            num_classes=num_classes, flow=flow, activation=activation,
            weights_path=aux2_weights_path, perm_invar=perm_invar, compute_acc=compute_acc)

    def forward(self, times, marks, masks, missing_masks=[], is_first_half=[]):
        with torch.inference_mode():
            aux1_output_dict = self.aux_model1.forward(times, marks, masks, missing_masks, is_first_half)
            aux2_output_dict = self.aux_model2.forward(times, marks, masks, missing_masks, is_first_half)
            aux1_output_dict = {
                'aux1_' + key: val for key, val in aux1_output_dict.items()}
            aux2_output_dict = {
                'aux2_' + key: val for key, val in aux2_output_dict.items()}

            aux1_output_dict.update(aux2_output_dict)
            #aux1_output_dict.update(
            #    {constants.ALPHA: self.alpha, constants.BETA: self.beta})

        output_dict = super().forward(times, marks, masks, missing_masks, is_first_half)
        output_dict.update(aux1_output_dict)
        return output_dict


class TransformerMixWithAux(TransformerMix):
    def __init__(self, name, activation, num_classes, d_model=256, d_inner=1024,
                 n_layers=2, cattn_n_layers=1, n_head=4, d_k=64, d_v=64, dropout=0.1,
                 attn_l=0, base_l=20, perm_invar=False, use_avg=True, share_weights=True,
                 attn_only=False, concat=False, num_latent=0, vi_method=None, num_z_samples=100,
                 compute_acc=True, aux1_weights_dir=None, aux2_weights_dir=None,
                 aux_lr=None, aux_weight_decay=None, aux_d_model=None):
                 #alpha_init=1.0, beta_init=0.0, affine_trainable=False):

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

        #if affine_trainable:
        #    self.alpha = nn.Parameter(torch.tensor(alpha_init))
        #    self.beta = nn.Parameter(torch.tensor(beta_init))
        #else:
        #    self.alpha = torch.tensor(alpha_init)
        #    self.beta = torch.tensor(beta_init)


        aux1_weights_path = load_checkpoint_path(aux1_weights_dir)
        aux2_weights_path = load_checkpoint_path(aux2_weights_dir)

        self.aux_model1 = TransformerMix(
            name=name, activation=activation, num_classes=num_classes, d_model=aux_d_model,
            d_inner=d_inner, n_layers=n_layers, cattn_n_layers=cattn_n_layers, n_head=n_head,
            d_k=d_k, d_v=d_v, dropout=dropout, attn_l=attn_l, base_l=base_l, perm_invar=perm_invar,
            use_avg=use_avg, share_weights=share_weights, attn_only=attn_only, concat=concat,
            num_latent=num_latent, vi_method=vi_method, num_z_samples=num_z_samples,
            compute_acc=compute_acc, weights_path=aux1_weights_path)

        self.aux_model2 = TransformerMix(
            name=name, activation=activation, num_classes=num_classes, d_model=aux_d_model,
            d_inner=d_inner, n_layers=n_layers, cattn_n_layers=cattn_n_layers, n_head=n_head,
            d_k=d_k, d_v=d_v, dropout=dropout, attn_l=attn_l, base_l=base_l, perm_invar=perm_invar,
            use_avg=use_avg, share_weights=share_weights, attn_only=attn_only, concat=concat,
            num_latent=num_latent, vi_method=vi_method, num_z_samples=num_z_samples,
            compute_acc=compute_acc, weights_path=aux2_weights_path)

    def forward(self, times, marks, masks, missing_masks=[], is_first_half=[]):
        with torch.inference_mode():
            aux1_output_dict = self.aux_model1.forward(times, marks, masks, missing_masks, is_first_half)
            aux2_output_dict = self.aux_model2.forward(times, marks, masks, missing_masks, is_first_half)
            aux1_output_dict = {
                'aux1_' + key: val for key, val in aux1_output_dict.items()}
            aux2_output_dict = {
                'aux2_' + key: val for key, val in aux2_output_dict.items()}

            aux1_output_dict.update(aux2_output_dict)
            #aux1_output_dict.update(
            #    {constants.ALPHA: self.alpha, constants.BETA: self.beta})

        output_dict = super().forward(times, marks, masks, missing_masks, is_first_half)
        output_dict.update(aux1_output_dict)
        return output_dict


