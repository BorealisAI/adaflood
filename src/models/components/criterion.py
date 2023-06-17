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
        flood_level = losses.shape[0] * self.flood_level
        adjusted_loss = (torch.sum(losses) - flood_level).abs() + flood_level

        return {constants.LOSS: adjusted_loss}


class IFloodTPPLoss(TPPLoss):
    def __init__(self, num_classes: int, flood_level: float = 0.0):
        super().__init__(num_classes)
        self.flood_level = flood_level

    def forward(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        losses = self.common_step(output_dict, input_dict)
        flood_level = losses.shape[0] * self.flood_level
        adjusted_loss = (losses - flood_level).abs() + flood_level

        return {constants.LOSS: adjusted_loss}


class AFloodTPPLoss(TPPLoss):
    def __init__(self, num_classes: int, alpha: float, gamma: float):
        super().__init__(num_classes)
        self.alpha = alpha
        self.gamma = gamma
        #self.weight_decay = weight_decay
        #self.exclude_layer_keywords = exclude_layer_keywords
        #self.kl_weight = kl_weight
        #self.gamma = gamma

        #self.tpp_loss = TPPLoss(
        #    weight_decay=weight_decay,
        #    exclude_layer_keywords=exclude_layer_keywords,
        #    kl_weight=kl_weight,
        #)

    def forward(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:

        model_output_names = [
            constants.HISTORIES,
            constants.EVENT_LL,
            constants.SURV_LL,
            constants.KL,
            constants.TIME_PREDS,
            constants.CLS_PREDS,
            constants.ATTENTIONS,
            constants.ENGINE,
        ]

        # Get main and aux model outputs
        model_outputs = {key: output_dict[key] for key in model_output_names}
        aux_model_outputs = {
            key: output_dict[f"aux_{key}"] for key in model_output_names
        }

        alpha = output_dict[constants.ALPHA]
        beta = output_dict[constants.BETA]

        times, masks = input_dict[constants.TIMES], input_dict[constants.MASKS]

        event_ll, surv_ll, kl = (
            model_outputs[constants.EVENT_LL],
            model_outputs[constants.SURV_LL],
            model_outputs[constants.KL],
        )
        aux_event_ll, aux_surv_ll, aux_kl = (
            aux_model_outputs[constants.EVENT_LL],
            aux_model_outputs[constants.SURV_LL],
            aux_model_outputs[constants.KL],
        )

        # AFlood
        base_loss = event_ll + surv_ll
        aux_loss = alpha * (aux_event_ll + aux_surv_ll) + beta
        flooded_loss = torch.abs(base_loss - aux_loss) + aux_loss

        loss = -torch.sum(flooded_loss)

        if kl is not None:
            loss += self.kl_weight * kl

        if self.weight_decay != 0.0:
            weight_squared = 0.0
            engine = output_dict["ENGINE"]
            for name, params in engine.named_parameters():
                for exclude_layer_keyword in self.exclude_layer_keywords:
                    if exclude_layer_keyword in name:
                        break
                else:
                    weight_squared += torch.sum(params**2)
            weight_squared *= 0.5
            loss += self.weight_decay * weight_squared

        loss_dict = {
            constants.LOSS: loss,
            "base_" + constants.LOSS: -torch.sum(base_loss),
            "aux_" + constants.LOSS: -torch.sum(aux_loss),
        }
        return loss_dict
