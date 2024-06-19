import os
import pickle
import torch
import numpy as np
import torch.nn as nn
from torchvision import models

from src import constants
from src.models.cls.resnet import ResBase
from src.utils.utils import load_checkpoint_path

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

res_dict = {
    "resnet18":models.resnet18,
    "resnet34":models.resnet34,
    "resnet50":models.resnet50,
    "resnet101":models.resnet101,
    "resnet152":models.resnet152,
    "resnext50":models.resnext50_32x4d,
    "resnext101":models.resnext101_32x8d
}

class ResBaseAux(ResBase):
    def __init__(self, name="resnet18", num_classes=10, d_model=64,
                 aux_logit_path=None, aux_lr=None, aux_weight_decay=None,
                 aux_d_model=None, pretrained=False, smaller=False):
        super().__init__(name=name, num_classes=num_classes,
                         d_model=d_model, pretrained=pretrained, smaller=smaller)

        aux_eval_logit_path = aux_logit_path
        #aux_train_logit_path = aux_logit_path.replace('eval', 'train')

        aux_eval_loss_path = aux_eval_logit_path.replace('logits', 'losses')
        #aux_train_loss_path = aux_train_logit_path.replace('logits', 'losses')

        #assert os.path.exists(aux_eval_logit_path), f"aux_eval_logit_path: {aux_eval_logit_path} does not exist"
        #assert os.path.exists(aux_train_logit_path), f"aux_train_logit_path: {aux_train_logit_path} does not exist"
        assert os.path.exists(aux_eval_loss_path), f"aux_eval_loss_path: {aux_eval_loss_path} does not exist"
        #assert os.path.exists(aux_train_loss_path), f"aux_train_loss_path: {aux_train_loss_path} does not exist"

        self.eval_logit_dict = None
        if os.path.exists(aux_eval_logit_path):
            with open(aux_eval_logit_path, "rb") as f:
                try:
                    self.eval_logit_dict = pickle.load(f)
                except:
                    self.eval_logit_dict = None

        #with open(aux_train_logit_path, "rb") as f:
        #    self.train_logit_dict = pickle.load(f)

        with open(aux_eval_loss_path, "rb") as f:
            self.eval_loss_dict = pickle.load(f)

        #with open(aux_train_loss_path, "rb") as f:
        #    self.train_loss_dict = pickle.load(f)

    def forward(self, input_dict):
        aux_output_dict = {}
        if self.training:
            orig_indices = input_dict[constants.INDICES]

            np_aux_eval_losses = []
            for idx in orig_indices:
                if idx.item() not in self.eval_loss_dict:
                    np_aux_eval_losses.append(0.)
                else:
                    np_aux_eval_losses.append(self.eval_loss_dict[idx.item()])
            np_aux_eval_losses = np.stack(np_aux_eval_losses)

            #np_aux_eval_losses = np.stack(
            #    [self.eval_loss_dict[idx.item()] for idx in orig_indices], axis=0)
            aux_eval_losses = torch.from_numpy(np_aux_eval_losses).to(
                input_dict[constants.IMAGES].device)
            aux_output_dict.update({constants.AUX_EVAL_LOSSES: aux_eval_losses})

            if self.eval_logit_dict is not None:
                dummy_logits = np.zeros(self.num_classes, dtype=float)
                np_aux_eval_logits = []
                for idx in orig_indices:
                    if idx.item() not in self.eval_logit_dict:
                        np_aux_eval_logits.append(dummy_logits)
                    else:
                        np_aux_eval_logits.append(self.eval_logit_dict[idx.item()])
                        #if np.any(np.isnan(self.eval_logit_dict[idx.item()])):
                        #    import IPython; IPython.embed(); exit()

                        #print('not dummy logits')
                np_aux_eval_logits = np.stack(np_aux_eval_logits)

                #np_aux_eval_logits = np.stack(
                #    [self.eval_logit_dict[idx.item()] for idx in orig_indices], axis=0)
                aux_eval_logits = torch.from_numpy(np_aux_eval_logits).to(
                    input_dict[constants.IMAGES].device)
                aux_output_dict.update({constants.AUX_LOGITS: aux_eval_logits})

            #np_aux_train_losses = np.stack(
            #    [self.train_loss_dict[idx.item()] for idx in orig_indices], axis=0)
            #aux_train_losses = torch.from_numpy(np_aux_train_losses).to(
            #    input_dict[constants.IMAGES].device)
            #aux_output_dict.update({constants.AUX_TRAIN_LOSSES: aux_train_losses})

        output_dict = super().forward(input_dict)
        output_dict.update(aux_output_dict)
        return output_dict

    #def forward(self, input_dict):
    #    with torch.inference_mode():
    #        aux1_output_dict = self.aux_model1.forward(input_dict)
    #        aux2_output_dict = self.aux_model2.forward(input_dict)
    #        aux1_output_dict = {
    #            'aux1_' + key: val for key, val in aux1_output_dict.items()}
    #        aux2_output_dict = {
    #            'aux2_' + key: val for key, val in aux2_output_dict.items()}

    #        aux1_output_dict.update(aux2_output_dict)

    #    output_dict = super().forward(input_dict)
    #    output_dict.update(aux1_output_dict)
    #    return output_dict

