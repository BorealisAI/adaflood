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

