import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(data_dir, split, transform=None) :
    is_train = split == 'train'
    root = os.path.join(data_dir, 'Animal10N', 'train' if is_train else 'test')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(split, input_size=224) :
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    is_train = split == 'train'
    resize_im = input_size > 32
    if is_train :
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
        if not resize_im :
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform

    # eval transform
    t = []
    if input_size <= 224 :
        crop_pct = 224 / 256
    else :
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

