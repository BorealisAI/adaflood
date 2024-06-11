# coding=utf-8

# pylint: skip-file
"""TDDPM model.
"""
import torch
import math
import torch.nn as nn
import functools

from lightning import LightningModule

from . import utils, layers, normalization
default_initializer = layers.default_init
get_act = layers.get_act

from src.models.diffuser.models.thp.models import TransformerEncoder

#RefineBlock = layers.RefineBlock
#ResidualBlock = layers.ResidualBlock
#ResnetBlockDDPM = layers.ResnetBlockDDPM
#Upsample = layers.Upsample
#Downsample = layers.Downsample
#conv3x3 = layers.ddpm_conv3x3
#get_act = layers.get_act
#get_normalization = normalization.get_normalization

from src import constants
from src.models.diffuser.models.thp.layers import EncoderLayer, ConditionedEncoderLayer

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(constants.PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq, diagonal=1):
    """ For masking out the subsequent info, i.e., masked self-attention. """
    sz_b, len_s, _ = seq.size() # seq: (B, T, D)
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=diagonal)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


#class TransformerEncoder(nn.Module):
#    """ A encoder model with self attention mechanism. """
#
#    def __init__(
#            self,
#            d_model, d_inner,
#            n_layers, n_head, d_k, d_v, dropout):
#        super().__init__()
#
#        self.d_model = d_model
#
#        # position vector, used for temporal encoding
#        self.position_vec = torch.tensor(
#            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
#            device=torch.device('cuda'))
#
#        # event type embedding
#        #self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=constants.PAD)
#
#        self.layer_stack = nn.ModuleList([
#            EncoderLayer(d_model, d_inner, n_head, d_k, d_v,
#                         dropout=dropout, normalize_before=False)
#            for _ in range(n_layers)])
#
#    def temporal_enc(self, time, non_pad_mask):
#        """
#        Input: batch*seq_len.
#        Output: batch*seq_len*d_model.
#        """
#        result = time.unsqueeze(-1) / self.position_vec
#        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
#        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
#        return result * non_pad_mask
#
#    def forward(self, features, masks):
#        """ Encode event sequences via masked self-attention. """
#
#        # TODO: add time embeddings
#        # prepare attention masks
#        # slf_attn_mask is where we cannot look, i.e., the future and the padding
#        slf_attn_mask_subseq = get_subsequent_mask(marks)
#
#        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=marks, seq_q=marks)
#        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
#        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
#
#        enc_output = self.temporal_enc(features, masks)
#        #enc_output = self.event_emb(marks) # (B, Seq, d_model)
#
#        for enc_layer in self.layer_stack:
#            #enc_output += tem_enc
#            enc_output, _ = enc_layer(
#                enc_output,
#                non_pad_mask=masks,
#                slf_attn_mask=slf_attn_mask)
#        return enc_output, None

@utils.register_model(name='tddpm')
class TDDPM(LightningModule):
  def __init__(self, config):
    super().__init__()

    self.config = config
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))
    self.scale_by_sigma = config.model.scale_by_sigma

    self.act = act = get_act(config)

    multiplier = self.config.model.multiplier
    d_model = self.d_model = self.config.data.height
    d_inner = self.config.data.height * multiplier
    n_layer = self.config.model.n_layer
    n_head=self.config.model.n_head
    d_k = self.config.data.height * multiplier
    d_v = self.config.data.height * multiplier
    dropout=self.config.model.dropout
    temb_dim = d_model * 4

    self.conditional = self.config.model.conditional

    # position vector, used for temporal encoding
    #self.position_vec = torch.tensor(
    #    [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
    #    device=torch.device('cuda'))

    self.layer_stack = nn.ModuleList([
        ConditionedEncoderLayer(d_model, d_inner, n_head, d_k, d_v,
                                dropout=dropout, normalize_before=True, temb_dim=temb_dim, act=act)
        if self.conditional else EncoderLayer(
            d_model, d_inner, n_head, d_k, d_v,
            dropout=dropout, normalize_before=True, temb_dim=temb_dim, act=act)
        for _ in range(n_layer)
    ])

    # Condition on noise levels.
    self.fc1_temb = nn.Linear(d_model, d_model * 4)
    self.fc1_temb.weight.data = default_initializer()(self.fc1_temb.weight.data.shape)
    nn.init.zeros_(self.fc1_temb.bias)
    self.fc2_temb = nn.Linear(d_model * 4, d_model * 4)
    self.fc2_temb.weight.data = default_initializer()(self.fc2_temb.weight.data.shape)
    nn.init.zeros_(self.fc2_temb.bias)

  def forward(self, x, labels, cond=None):
    # TODO: (sqeuence-wise) attention blocks - no need down and upsampling blocks?

    # timestep/scale embedding
    timesteps = labels
    temb = layers.get_timestep_embedding(timesteps, self.d_model)
    temb = self.fc1_temb(temb)
    temb = self.fc2_temb(self.act(temb))

    h = x.squeeze(1).transpose(1, 2) # (B, 1, D, T) -> (B, T, D)
    cond = cond.squeeze(1).transpose(1, 2) # (B, 1, D, T) -> (B, T, D)

    # prepare attention masks
    # slf_attn_mask is where we cannot look, i.e., the future and the padding
    slf_attn_mask = get_subsequent_mask(h).gt(0)

    for enc_layer in self.layer_stack:
      h, _ = enc_layer(h, slf_attn_mask=slf_attn_mask, temb=temb, cond=cond)

    if self.scale_by_sigma: # False for DDPM
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = self.sigmas[labels, None, None, None]
      h = h / used_sigmas

    h = h.unsqueeze(1).transpose(2, 3) # (B, T, D) -> (B, 1, D, T)

    return h

