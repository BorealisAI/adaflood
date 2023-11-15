""" This module defines losses for hierarchical contrastive loss """
import torch
import numpy as np
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

            #aux_min_loss = output_dict['aux_min_loss']
            #aux_event_losses = aux_min_loss
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


            # ==========================================================
            #with torch.no_grad():
            #    # time preds
            #    masks = input_dict[constants.MASKS][:,1:]
            #    times = input_dict[constants.TIMES][:,1:]
            #    time_preds = output_dict[constants.AUX_PREDS]

            #    inter_times = (1-self.gamma) * time_preds + self.gamma * times
            #    prob_dist = output_dict[constants.PROB_DIST]
            #    lognorm_dist = output_dict[constants.LOGNORM_DIST]
            #    log_weights = output_dict[constants.LOG_WEIGHTS]

            #    losses_preds = prob_dist.compute_losses_and_preds(
            #        lognorm_dist, log_weights, inter_times, masks)
            #    aux_event_losses = -(losses_preds['event_ll'] + losses_preds['surv_ll'])

            #    # cls losses
            #    if constants.AUX_LOGITS in output_dict:
            #        probs = torch.softmax(output_dict[constants.AUX_LOGITS], dim=-1)
            #        marks = input_dict[constants.MARKS][:,1:]
            #        adjusted_marks = torch.where(
            #            marks-1 >= 0, marks-1, torch.zeros_like(marks))

            #        batch_size, dim, num_classes = probs.shape
            #        flattened_probs, flattened_marks = (
            #            probs.reshape(-1, num_classes), adjusted_marks.reshape(-1))
            #        true_class_probs = flattened_probs[
            #            torch.arange(flattened_marks.shape[0]), flattened_marks]

            #        aux_cls_losses = -torch.log((1-self.gamma) * true_class_probs + self.gamma).reshape(
            #            batch_size, dim)
            #        aux_cls_losses = aux_cls_losses * masks.squeeze(-1)
            #        aux_cls_losses = torch.sum(aux_cls_losses, dim=-1)

            #        aux_losses = aux_event_losses + aux_cls_losses
            #    else:
            #        aux_losses = aux_event_losses
            # ==========================================================

            #import IPython; IPython.embed()
            #if constants.AUX_LOGITS in output_dict:
            #    probs = torch.softmax(output_dict[constants.AUX_LOGITS], dim=1)
            #    marks = input_dict[constants.MARKS][:,1:]
            #    adjusted_marks = torch.where(
            #        marks-1 >= 0, marks-1, torch.zeros_like(marks))

            #    batch_size, dim, num_classes = probs.shape
            #    flattened_probs, flattened_marks = (
            #        probs.reshape(-1, num_classes), adjusted_marks.reshape(-1))
            #    true_class_probs = flattened_probs[
            #        torch.arange(flattened_marks.shape[0]), flattened_marks]

            #    aux_cls_losses = torch.sum(
            #        -torch.log((1-self.gamma) * true_class_probs + self.gamma).reshape(
            #            batch_size, dim), dim=-1)
            #    aux_losses = aux_event_losses + aux_cls_losses
            #else:
            #    aux_losses = aux_event_losses

            trans_aux_losses = self.alpha * aux_losses + self.beta

            ## TODO: get the minimum of aux_losses and interpolate
            #aux_adjusted_losses = torch.where(
            #    losses >= trans_aux_losses,
            #    losses,
            #    trans_aux_losses)
            #    #torch.zeros_like(losses))

            aux_adjusted_losses = (losses - trans_aux_losses).abs() + trans_aux_losses
            adjusted_loss = torch.sum(aux_adjusted_losses)
        else:
            adjusted_loss = torch.sum(losses)

        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}


    #def forward(
    #    self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    #) -> Dict[str, torch.Tensor]:

    #    losses = self.common_step(output_dict, input_dict)
    #    if self.training:
    #        aux1_losses = self.aux_step(output_dict, input_dict, 'aux1')
    #        aux2_losses = self.aux_step(output_dict, input_dict, 'aux2')

    #        # compute loss based on first_half bool
    #        if len(input_dict['is_first_half']) > 0:
    #            is_first_half = input_dict['is_first_half']
    #            is_second_half = torch.logical_not(is_first_half)

    #            trans_aux1_losses = self.alpha * aux1_losses[is_second_half] + self.beta
    #            trans_aux2_losses = self.alpha * aux2_losses[is_first_half] + self.beta

    #            aux1_adjusted_losses = (
    #                losses[is_second_half] - trans_aux1_losses).abs() + trans_aux1_losses
    #            aux2_adjusted_losses = (
    #                losses[is_first_half] - trans_aux2_losses).abs() + trans_aux2_losses
    #            adjusted_loss = torch.sum(aux1_adjusted_losses) + torch.sum(aux2_adjusted_losses)
    #        else:
    #            adjusted_loss = torch.sum(losses)
    #    else:
    #        adjusted_loss = torch.sum(losses)

    #    return {constants.LOSS: adjusted_loss}


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
            # =====================================================
            ##aux_train_losses = torch.ones_like(aux_train_losses).to(aux_train_losses.device) * 0.35
            #aux_train_losses = torch.zeros_like(aux_train_losses).to(aux_train_losses.device)
            ##aux_train_losses= torch.where(
            ##    aux_eval_losses < aux_train_losses, aux_eval_losses, aux_train_losses)

            #aux_inter_losses = -torch.log(
            #    (1-self.gamma) * torch.exp(-aux_eval_losses) + self.gamma)

            #aux_adjusted_losses = torch.where(
            #    losses >= aux_inter_losses,
            #    losses,
            #    torch.where(losses <= aux_train_losses,
            #                2 * aux_train_losses - losses,
            #                torch.zeros_like(losses)))
            #adjusted_loss = torch.mean(aux_adjusted_losses)
            # =====================================================
            # =====================================================
            #aux_eval_losses = -torch.log(
            #    (1-self.gamma) * torch.exp(-aux_eval_losses) + self.gamma)
            #aux_train_losses = torch.zeros_like(aux_eval_losses).to(aux_eval_losses.device)

            #aux_adjusted_losses = torch.where(
            #    losses >= aux_eval_losses,
            #    losses,
            #    aux_train_losses)
            #adjusted_loss = torch.mean(aux_adjusted_losses)
            # =====================================================
            # =====================================================
            #aux_inter_losses = self.gamma * aux_eval_losses +\
            #    (1 - self.gamma) * aux_train_losses
            #aux_adjusted_losses = torch.where(
            #    losses >= aux_inter_losses,
            #    losses,
            #    torch.where(losses <= aux_train_losses,
            #                2 * aux_train_losses - losses,
            #                torch.zeros_like(losses))
            #)
            #adjusted_loss = torch.mean(aux_adjusted_losses)
            # =====================================================

            # =====================================================
            trans_aux_losses = self.alpha * aux_eval_losses + self.beta

            trans_aux_losses = -torch.log(
                (1-self.gamma) * torch.exp(-trans_aux_losses) + self.gamma)

            #trans_aux_losses = torch.where(
            #    trans_aux_losses > self.upper_bound,
            #    torch.ones_like(trans_aux_losses) * self.flood_level, trans_aux_losses)

            aux_adjusted_losses = (
                losses - trans_aux_losses).abs() + trans_aux_losses
            adjusted_loss = torch.mean(aux_adjusted_losses)
            # =====================================================
        else:
            adjusted_loss = torch.mean(losses)

        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}


class DistillCLSLoss(nn.Module):
    def __init__(self, num_classes: int, distill_weight: float):
        super(DistillCLSLoss, self).__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.distill_weight = distill_weight

    def common_step(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        logits, labels = output_dict[constants.LOGITS], input_dict[constants.LABELS]
        losses = self.ce(logits, labels)
        return losses

    def distill_step(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        teacher_logits = output_dict[constants.AUX_LOGITS]
        teacher_probs = torch.nn.functional.softmax(teacher_logits / self.distill_weight , dim=-1)

        logits = output_dict[constants.LOGITS]
        log_probs = torch.nn.functional.log_softmax(logits / self.distill_weight, dim=-1)

        soft_loss = -torch.sum(teacher_probs * log_probs) / log_probs.size()[0]

        return soft_loss

    def forward(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) ->  Dict[str, torch.Tensor]:
        losses = self.common_step(output_dict, input_dict)
        if self.training:
            soft_loss = self.distill_step(output_dict, input_dict)
            ce_loss = torch.mean(losses)
            adjusted_loss = ce_loss + (1.0 / self.distill_weight ** 2) * soft_loss
        else:
            adjusted_loss = torch.mean(losses)
        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}



    #def forward(
    #    self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    #) -> Dict[str, torch.Tensor]:

    #    losses = self.common_step(output_dict, input_dict)
    #    if self.training:
    #        aux1_losses = self.aux_step(output_dict, input_dict, 'aux1')
    #        aux2_losses = self.aux_step(output_dict, input_dict, 'aux2')

    #        # compute loss based on first_half bool
    #        if len(input_dict['is_first_half']) > 0:
    #            is_first_half = input_dict['is_first_half']
    #            is_second_half = torch.logical_not(is_first_half)

    #            trans_aux1_losses = self.alpha * aux1_losses[is_second_half] + self.beta
    #            trans_aux2_losses = self.alpha * aux2_losses[is_first_half] + self.beta

    #            aux1_adjusted_losses = (
    #                losses[is_second_half] - trans_aux1_losses).abs() + trans_aux1_losses
    #            aux2_adjusted_losses = (
    #                losses[is_first_half] - trans_aux2_losses).abs() + trans_aux2_losses
    #            adjusted_loss = torch.mean(aux1_adjusted_losses) + torch.mean(aux2_adjusted_losses)
    #        else:
    #            adjusted_loss = torch.mean(losses)
    #    else:
    #        adjusted_loss = torch.mean(losses)

    #    return {constants.LOSS: adjusted_loss}


