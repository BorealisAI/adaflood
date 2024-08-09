# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Dict, Union

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
        return {constants.LOSS: torch.sum(losses), constants.LOSSES: losses}


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
        else:
            adjusted_loss = torch.sum(losses)
        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}


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
        else:
            adjusted_loss = torch.sum(losses)
        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}


class AdaFloodTPPLoss(TPPLoss):
    def __init__(self, num_classes: int, alpha_init: float = 1.0,
                 beta_init: float = 0.0, affine_train: str = None, gamma: float = None):
        super().__init__(num_classes)

        if affine_train is None:
            self.alpha = torch.tensor(alpha_init)
            self.beta = torch.tensor(beta_init)
        elif affine_train == 'alpha':
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
            self.beta = torch.tensor(beta_init)
        elif affine_train == 'beta':
            self.alpha = torch.tensor(alpha_init)
            self.beta = nn.Parameter(torch.tensor(beta_init))
        elif affine_train == 'both':
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
            self.beta = nn.Parameter(torch.tensor(beta_init))
        else:
            raise NotImplementedError(f'affine_train: {affine_train} is not implemented')

        self.gamma = gamma

    def aux_step(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict, aux_type: str
    ) -> Dict[str, torch.Tensor]:
        return output_dict[constants.AUX_LOSSES]

    def forward(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:

        losses = self.common_step(output_dict, input_dict)
        if self.training:
            times = input_dict[constants.TIMES][:,1:]
            masks = input_dict[constants.MASKS][:,1:]
            aux_log_weights = output_dict[constants.AUX_LOG_WEIGHTS]
            aux_prob_dist = output_dict[constants.AUX_PROB_DIST]

            prob_dist = output_dict[constants.PROB_DIST]
            losses_preds = prob_dist.compute_losses_and_preds(
                aux_prob_dist, aux_log_weights, times, masks)
            aux_event_losses = -(losses_preds['event_ll'] + losses_preds['surv_ll'])
            aux_event_losses = (1-self.gamma) * aux_event_losses

            # cls losses
            if constants.AUX_LOGITS in output_dict:
                probs = torch.softmax(output_dict[constants.AUX_LOGITS], dim=-1)
                marks = input_dict[constants.MARKS][:,1:]
                adjusted_marks = torch.where(
                    marks-1 >= 0, marks-1, torch.zeros_like(marks))

                batch_size, dim, num_classes = probs.shape
                flattened_probs, flattened_marks = (
                    probs.reshape(-1, num_classes), adjusted_marks.reshape(-1))
                true_class_probs = flattened_probs[
                    torch.arange(flattened_marks.shape[0]), flattened_marks]

                aux_cls_losses = -torch.log((1-self.gamma) * true_class_probs + self.gamma).reshape(
                    batch_size, dim)
                aux_cls_losses = aux_cls_losses * masks.squeeze(-1)
                aux_cls_losses = torch.sum(aux_cls_losses, dim=-1)

                aux_losses = aux_event_losses + aux_cls_losses
            else:
                aux_losses = aux_event_losses


            trans_aux_losses = self.alpha * aux_losses + self.beta

            aux_adjusted_losses = (losses - trans_aux_losses).abs() + trans_aux_losses
            adjusted_loss = torch.sum(aux_adjusted_losses)
        else:
            adjusted_loss = torch.sum(losses)

        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}


class CLSLoss(nn.Module):
    def __init__(self, num_classes: int):
        super(CLSLoss, self).__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def common_step(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        logits, labels = output_dict[constants.LOGITS], input_dict[constants.LABELS]
        losses = self.ce(logits, labels)
        return losses

    def forward(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) ->  Dict[str, torch.Tensor]:
        losses = self.common_step(output_dict, input_dict)
        return {constants.LOSS: torch.mean(losses), constants.LOSSES: losses}


class FloodCLSLoss(CLSLoss):
    def __init__(self, num_classes: int, flood_level: float = 0.0):
        super().__init__(num_classes)
        self.flood_level = flood_level

    def forward(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        losses = self.common_step(output_dict, input_dict)
        if self.training:
            adjusted_loss = (torch.mean(losses) - self.flood_level).abs() + self.flood_level
            #import IPython; IPython.embed()
        else:
            adjusted_loss = torch.mean(losses)
        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}


class IFloodCLSLoss(CLSLoss):
    def __init__(self, num_classes: int, flood_level: float = 0.0):
        super().__init__(num_classes)
        self.flood_level = flood_level

    def forward(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        losses = self.common_step(output_dict, input_dict)
        if self.training:
            adjusted_loss = torch.mean((losses - self.flood_level).abs() + self.flood_level)
        else:
            adjusted_loss = torch.mean(losses)
        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}


class AdaFloodCLSLoss(CLSLoss):
    def __init__(self, num_classes: int, alpha_init: float = 1.0,
                 beta_init: float = 0.0, affine_train: str = None,
                 gamma: float = 0.5):
        super().__init__(num_classes)

        if affine_train is None:
            self.alpha = torch.tensor(alpha_init)
            self.beta = torch.tensor(beta_init)
        elif affine_train == 'alpha':
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
            self.beta = torch.tensor(beta_init)
        elif affine_train == 'beta':
            self.alpha = torch.tensor(alpha_init)
            self.beta = nn.Parameter(torch.tensor(beta_init))
        elif affine_train == 'both':
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
            self.beta = nn.Parameter(torch.tensor(beta_init))
        else:
            raise NotImplementedError(f'affine_train: {affine_train} is not implemented')

        self.gamma = gamma

    def aux_step(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict, aux_type: str
    ) -> Dict[str, torch.Tensor]:
        eval_losses = output_dict[constants.AUX_EVAL_LOSSES]
        return eval_losses

    def forward(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:

        losses = self.common_step(output_dict, input_dict)
        if self.training:
            #probs = torch.softmax(output_dict[constants.AUX_LOGITS], dim=1)
            aux_eval_losses = self.aux_step(
                output_dict, input_dict, 'aux')
            
            trans_aux_losses = self.alpha * aux_eval_losses + self.beta

            trans_aux_losses = -torch.log(
                (1-self.gamma) * torch.exp(-trans_aux_losses) + self.gamma)

            aux_adjusted_losses = (
                losses - trans_aux_losses).abs() + trans_aux_losses
            adjusted_loss = torch.mean(aux_adjusted_losses)
        else:
            adjusted_loss = torch.mean(losses)

        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}

