import torch
import torch.nn as nn
#from torchvision import models

from src import constants
from src.models.cls import resnet_source

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
    "resnet18": resnet_source.resnet18,
    "resnet34": resnet_source.resnet34,
    "resnet50": resnet_source.resnet50,
    #"resnet101": resnet_source.resnet101,
    #"resnet152": resnet_source.resnet152,
}

class ResBase(nn.Module):
    def __init__(self, name: str = "resnet18", num_classes: int = 10,
                 d_model: int = 64, weights_path: str = None, pretrained=False, smaller=False):
        super().__init__()
        model_resnet = res_dict[name](d_model=d_model, pretrained=pretrained, smaller=smaller)
        self.use_maxpool = False
        # feature extractor
        self.smaller = smaller
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        if self.use_maxpool:
            self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        if not self.smaller:
            self.layer4 = model_resnet.layer4

        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        # classifier
        self.num_classes = num_classes
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.in_features, num_classes)
        self.fc.apply(init_weights)

        if weights_path is not None:
            checkpoint = torch.load(weights_path)['state_dict']
            try:
                self.load_state_dict(checkpoint)
            except:
                checkpoint = {key.replace("net.", "", 1): value for key, value in checkpoint.items() if key.startswith("net.")}
                self.load_state_dict(checkpoint)

    def forward(self, input_dict):
        x = input_dict[constants.IMAGES]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.use_maxpool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if not self.smaller:
            x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output_dict = {constants.LOGITS: x}
        return output_dict

