""" This module defines losses for hierarchical contrastive loss """
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List, Dict, Optional, Union

from src import constants


class TPPLoss(nn.Module):
    def __init__(self, num_classes: int):
        super(TPPLoss, self).__init__()
        self.num_classes = num_classes

    def common_step(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        # compute nll
        event_ll, surv_ll, kl, cls_ll= (
            output_dict[constants.EVENT_LL], output_dict[constants.SURV_LL],
            output_dict[constants.KL], output_dict[constants.CLS_LL])
        losses = -(event_ll + surv_ll)

        if cls_ll is not None:
            losses += -cls_ll # NOTE: negative ll

        if kl is not None:
            losses += kl

        return losses

    def forward(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) ->  Dict[str, torch.Tensor]:
        losses = self.common_step(output_dict, input_dict)
        return {constants.LOSS: torch.sum(losses)}


class FloodTPPLoss(TPPLoss):
    def __init__(self, num_classes: int, flood_level: float = 0.0):
        super().__init__(num_classes)
        self.flood_level = flood_level

    def forward(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        losses = self.common_step(output_dict, input_dict)
        if self.training:
            masks = input_dict[constants.MASKS].bool()
            flood_level = masks.sum() * self.flood_level
            adjusted_loss = (torch.sum(losses) - flood_level).abs() + flood_level
            #adjusted_loss = torch.sum(losses)
        else:
            adjusted_loss = torch.sum(losses)
        return {constants.LOSS: adjusted_loss}


class IFloodTPPLoss(TPPLoss):
    def __init__(self, num_classes: int, flood_level: float = 0.0):
        super().__init__(num_classes)
        self.flood_level = flood_level

    def forward(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        losses = self.common_step(output_dict, input_dict)
        if self.training:
            masks = input_dict[constants.MASKS].bool()
            flood_level = masks.sum(dim=1).view(-1) * self.flood_level
            adjusted_loss = torch.sum((losses - flood_level).abs() + flood_level)
            #adjusted_loss = torch.sum(losses)
        else:
            adjusted_loss = torch.sum(losses)
        return {constants.LOSS: adjusted_loss}


class AdaFloodTPPLoss(TPPLoss):
    def __init__(self, num_classes: int, alpha_init: float = 1.0,
                 beta_init: float = 0.0, affine_trainable: bool = False):
        super().__init__(num_classes)

        if affine_trainable:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
            self.beta = nn.Parameter(torch.tensor(beta_init))
        else:
            self.alpha = torch.tensor(alpha_init)
            self.beta = torch.tensor(beta_init)

    def aux_step(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict, aux_type: str
    ) -> Dict[str, torch.Tensor]:
        # compute nll
        event_ll, surv_ll, kl, cls_ll= (
            output_dict[aux_type + '_' + constants.EVENT_LL],
            output_dict[aux_type + '_' + constants.SURV_LL],
            output_dict[aux_type + '_' + constants.KL],
            output_dict[aux_type + '_' + constants.CLS_LL])
        losses = -(event_ll + surv_ll)

        if cls_ll is not None:
            losses += -cls_ll # NOTE: negative ll

        if kl is not None:
            losses += kl

        return losses

    def forward(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:

        losses = self.common_step(output_dict, input_dict)
        if self.training:
            aux1_losses = self.aux_step(output_dict, input_dict, 'aux1')
            aux2_losses = self.aux_step(output_dict, input_dict, 'aux2')

            # compute loss based on first_half bool
            if len(input_dict['is_first_half']) > 0:
                is_first_half = input_dict['is_first_half']
                is_second_half = torch.logical_not(is_first_half)

                trans_aux1_losses = self.alpha * aux1_losses[is_second_half] + self.beta
                trans_aux2_losses = self.alpha * aux2_losses[is_first_half] + self.beta

                aux1_adjusted_losses = (
                    losses[is_second_half] - trans_aux1_losses).abs() + trans_aux1_losses
                aux2_adjusted_losses = (
                    losses[is_first_half] - trans_aux2_losses).abs() + trans_aux2_losses
                adjusted_loss = torch.sum(aux1_adjusted_losses) + torch.sum(aux2_adjusted_losses)
            else:
                adjusted_loss = torch.sum(losses)
        else:
            adjusted_loss = torch.sum(losses)

        return {constants.LOSS: adjusted_loss}

