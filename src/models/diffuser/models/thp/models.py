import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import constants
from src.models.diffuser.models.thp.layers import EncoderLayer

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
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=diagonal)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

def get_l_subsequent_mask(seq, l, unmask_offset=0):
    """ For masking out the subsequent info, i.e., masked self-attention. """
    sz_b, len_s = seq.size()
    subsequent_mask = torch.ones(
        (len_s, len_s), device=seq.device, dtype=torch.uint8)

    for idx in range(len_s):
        subsequent_mask[idx][max(idx-l+1, 0):idx+unmask_offset+1] = 0

    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask



class TransformerEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        #self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v,
                         dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, x, labels): #, masks):
        """ Encode event sequences via masked self-attention. """

        # TODO: add time embeddings
        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(marks)

        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=marks, seq_q=marks)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        enc_output = self.temporal_enc(features, masks)
        #enc_output = self.event_emb(marks) # (B, Seq, d_model)

        for enc_layer in self.layer_stack:
            #enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=masks,
                slf_attn_mask=slf_attn_mask)
        return enc_output, None
