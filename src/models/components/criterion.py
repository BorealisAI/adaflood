import torch
import torch.nn as nn
from typing import Dict, Union

from src import constants

class TPPLoss(nn.Module):
    def __init__(self, num_classes: int, kl_weight=1e-5):
        super(TPPLoss, self).__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight

    def compute_nlls(
        self, output_dict: Union[Dict, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # compute nll
        event_ll, surv_ll, cls_ll= (
            output_dict[constants.EVENT_LL], output_dict[constants.SURV_LL],
            output_dict[constants.CLS_LL])

        if surv_ll is None:
            nlls = -event_ll
        else:
            nlls = -(event_ll + surv_ll)

        if cls_ll is not None:
            nlls += -cls_ll # NOTE: negative ll

        output_dict[constants.NLLS] = nlls
        return nlls


    def common_step(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        # compute nll
        event_ll, surv_ll, kl, cls_ll= (
            output_dict[constants.EVENT_LL], output_dict[constants.SURV_LL],
            output_dict[constants.KL], output_dict[constants.CLS_LL])
        if surv_ll is None:
            losses = -event_ll
        else:
            losses = -(event_ll + surv_ll)

        if cls_ll is not None:
            losses += -cls_ll # NOTE: negative ll

        if kl is not None:
            losses += self.kl_weight * kl

        return losses

    def forward(
        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
    ) ->  Dict[str, torch.Tensor]:
        nlls = self.compute_nlls(output_dict)

        kl = output_dict[constants.KL]
        if kl is not None:
            losses = nlls + self.kl_weight * kl
        else:
            losses = nlls

        #losses = self.common_step(output_dict, input_dict)
        return {constants.LOSS: torch.sum(losses), constants.LOSSES: losses}

