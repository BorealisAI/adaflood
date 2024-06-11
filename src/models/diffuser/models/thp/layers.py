import torch.nn as nn
import torch
import numpy as np

from src.models.tpp.thp.sublayers import (
    MultiHeadAttention, PositionwiseFeedForward)


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init


def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1,
                 temb_dim=None, normalize_before=True, act=None):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

        if temb_dim is not None:
            self.fc_temb = nn.Linear(temb_dim, d_model)
            self.fc_temb.weight.data = default_init()(self.fc_temb.weight.data.shape)
            nn.init.zeros_(self.fc_temb.bias)
            self.act = act
            assert self.act is not None

    def forward(self, enc_input, slf_attn_mask=None, temb=None, cond=None):
        # enc_input as query and cond as key and value
        if cond is None:
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input, mask=slf_attn_mask)
        else:
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, cond, cond, mask=slf_attn_mask)

        if temb is not None:
            res_temb = self.fc_temb(self.act(temb))[:, None, :] # (B, D) -> (B, 1, D)
            enc_output += res_temb

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn

class ConditionedEncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1,
                 temb_dim=None, normalize_before=True, act=None):
        super(ConditionedEncoderLayer, self).__init__()
        self.act = act
        self.enc_in_layer = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            self.act,
            nn.Linear(d_model, d_model))

        self.enc_temb_out_layer = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model))

        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.cross_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

        if temb_dim is not None:
            self.fc_temb = nn.Linear(temb_dim, d_model)
            self.fc_temb.weight.data = default_init()(self.fc_temb.weight.data.shape)
            nn.init.zeros_(self.fc_temb.bias)
            assert self.act is not None

    def forward(self, enc_input, slf_attn_mask=None, temb=None, cond=None):
        # enc_input and temb

        ## process enc_input
        enc_output = self.enc_in_layer(enc_input)

        ## process temb
        temb = self.fc_temb(self.act(temb))[:, None, :] # (B, D) -> (B, 1, D)

        ## residual connect between enc_output and temb
        enc_output = enc_output + temb

        ## process enc_output + temb
        enc_output = self.enc_temb_out_layer(enc_output)


        # enc_output and cond

        ## self-attention enc_output
        slf_attn_output, enc_slf_attn = self.slf_attn(
            enc_output, enc_output, enc_output, mask=slf_attn_mask)

        ## residual connect
        enc_output = enc_output + slf_attn_output

        ## cross-attention between enc_output and cond
        assert cond is not None
        cross_attn_output, enc_cross_attn = self.cross_attn(
                enc_input, cond, cond, mask=slf_attn_mask)

        ## residual connect between enc_output and cross_attn
        enc_output = enc_output + cross_attn_output

        # last feed forward layers
        enc_output = self.pos_ffn(enc_output) # TODO: remove residual connection

        return enc_output, (enc_slf_attn, enc_cross_attn)

