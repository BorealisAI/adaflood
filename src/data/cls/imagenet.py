import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class ImageNet100(Dataset):
    def __init__(self, data_dir, split, transform=None):
        assert split in ['train', 'val', 'test']
        self.data_dir = data_dir
        self.img_path = []
        self.labels = []
        self.transform = transform

        if split in ['train', 'val']:
            imagenet_split = 'train'
        else:
            imagenet_split = 'val'

        # The class subset is taken from: https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
        with open(os.path.join(self.data_dir, 'splits', 'imagenet100.txt')) as f:
            class_names = list(map(lambda x : x.strip(), f.readlines()))

        num_classes = 100 # Subset of ImageNet
        for i, class_name in enumerate(class_names):
            folder_path = os.path.join(
                self.data_dir, 'imagenet', 'raw', imagenet_split, class_name)

            file_names = os.listdir(folder_path)

            if split is not 'test':
                num_train = int(len(file_names) * 0.8) # 80% Training data

            for j, fid in enumerate(file_names):
                if split == 'train' and j >= num_train: # ensures only the first 80% of data is used for training
                    break
                elif split == 'val' and j < num_train: # skips the first 80% of data used for training
                    continue
                self.img_path.append(os.path.join(folder_path, fid))
                self.labels.append(i)

        print(f"Dataset Size: {len(self.labels)}")
        self.targets = self.labels # Sampler needs to use targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

class ImageNet(Dataset):
    def __init__(self, data_dir, split, transform=None):
        assert split in ['train', 'val', 'test']
        self.data_dir = data_dir
        self.img_path = []
        self.labels = []
        self.transform = transform

        if split in ['train', 'val']:
            imagenet_split = 'train'
        else:
            imagenet_split = 'val'

        # The class subset is taken from: https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
        with open(os.path.join(self.data_dir, 'splits', 'imagenet.txt')) as f:
            class_names = list(map(lambda x : x.strip(), f.readlines()))

        for i, class_name in enumerate(class_names):
            folder_path = os.path.join(
                self.data_dir, 'imagenet', 'raw', imagenet_split, class_name)

            file_names = os.listdir(folder_path)

            if split is not 'test':
                num_train = int(len(file_names) * 0.8)

            for j, fid in enumerate(file_names):
                if split == 'train' and j >= num_train: # ensures only the first 80% of data is used for training
                    break
                elif split == 'val' and j < num_train: # skips the first 80% of data used for training
                    continue
                self.img_path.append(os.path.join(folder_path, fid))
                self.labels.append(i)

        print(f"Dataset Size: {len(self.labels)}")
        self.targets = self.labels # Sampler needs to use targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label
