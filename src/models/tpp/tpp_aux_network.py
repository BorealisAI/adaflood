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
                 aux1_weights_dir=None, aux2_weights_dir=None):
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

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        aux1_weights_path = load_checkpoint_path(aux1_weights_dir)
        aux2_weights_path = load_checkpoint_path(aux2_weights_dir)

        self.aux_model1 = IntensityFreePredictor(
            name=name, hidden_dim=hidden_dim, num_components=num_components,
            num_classes=num_classes, flow=flow, activation=activation,
            weights_path=aux1_weights_path, perm_invar=perm_invar, compute_acc=compute_acc)

        self.aux_model2 = IntensityFreePredictor(
            name=name, hidden_dim=hidden_dim, num_components=num_components,
            num_classes=num_classes, flow=flow, activation=activation,
            weights_path=aux2_weights_path, perm_invar=perm_invar, compute_acc=compute_acc)


    def forward(self, times, marks, masks, missing_masks=[], is_first_half=[]):
        # TODO: Don't we need two aux models? if so, which data goes to which aux models? Keep track
        # of which data each model was trained on?
        with torch.inference_mode():
            aux1_output_dict = self.aux_model1.forward(times, marks, masks, missing_masks, is_first_half)
            aux2_output_dict = self.aux_model2.forward(times, marks, masks, missing_masks, is_first_half)
            aux1_output_dict = {
                'aux1_' + key: val for key, val in aux1_output_dict.items()}
            aux2_output_dict = {
                'aux2_' + key: val for key, val in aux2_output_dict.items()}

            aux1_output_dict.update(aux2_output_dict)
            aux1_output_dict.update(
                {constants.ALPHA: self.alpha, constants.BETA: self.beta})

        output_dict = super().forward(times, marks, masks, missing_masks, is_first_half)
        output_dict.update(aux1_output_dict)
        return output_dict



class TransformerMix(LightningModule):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, name, activation, num_classes, d_model=256, d_inner=1024,
                 n_layers=2, cattn_n_layers=1, n_head=4, d_k=64, d_v=64, dropout=0.1,
                 attn_l=0, base_l=20, perm_invar=False, use_avg=True, share_weights=True,
                 attn_only=False, concat=False, num_latent=0, vi_method=None,
                 num_z_samples=100, compute_acc=True):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.compute_acc = compute_acc

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

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_classes)

        # prediction of next time stamp
        self.time_predictor = TransformerDecoder(d_model, 1)

        # prediction of next event type
        self.class_predictor = TransformerDecoder(d_model, num_classes)
        self.class_predictor.linear = self.linear

        self.mark_linear = nn.Linear(d_model, self.num_classes)

        self.activation = util.build_activation(activation)
        self.prob_dist = LogNormalMixture(
            d_model, components=8, activation=self.activation, vi_method=self.vi_method)

        trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The number of trainable model parameters: {trainable_params}', flush=True)


    def forward(self, times, marks, masks, missing_masks=[]):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_{N-1}), we predict (l_2, ..., l_N).
        Input: times: (B, Seq, 1);
               marks: (B, Seq, 1);
               masks: (B, Seq, 1).
        """
        batch_size = times.shape[0]
        times = times.squeeze(-1)
        marks = marks.squeeze(-1)

        # obtain the features from the transformer encoder
        encode_out, _ = self.encoder(times, marks, masks)

        # compute the global feature G
        if self.perm_invar:
            target_inputs = encode_out
            zeros = torch.zeros((encode_out.shape[0], 1, encode_out.shape[2])).to(times)
            context_encode = torch.cat((zeros, encode_out[:,:-1]), dim=1)
            context_encode = torch.cumsum(context_encode, dim=1)
            if self.use_avg:
                num_cum_seq = torch.arange(1, context_encode.shape[1]+1).reshape(1, -1, 1).to(times)
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
        attentions = None
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

        # obatin log-likelihood from the mixture of log-normals
        prob_output_dict = self.prob_dist(
            histories, times[:,1:], masks[:,1:]) # (B, Seq-1, 1): ignore the first event since that's only input not output
        event_ll = prob_output_dict['event_ll']
        surv_ll = prob_output_dict['surv_ll']
        time_predictions = prob_output_dict['preds']

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
            constants.KL: kl,
            constants.TIME_PREDS: time_predictions,
            constants.CLS_LL: class_log_probs,
            constants.CLS_LOGITS: class_logits,
            constants.CLS_PREDS: class_predictions,
            constants.ATTENTIONS: attentions,
        }
        return output_dict

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var, masks):
        kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = kl_div * masks[:,:-1]
        kl_div = 0.5 * kl_div.sum(dim=(1, 2))
        return kl_div

