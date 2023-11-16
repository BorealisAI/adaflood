import os
import os.path as osp
import pickle
import PIL
from PIL import Image
import numpy as np
import lmdb
import six
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as T
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


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

            if split != 'test':
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

#class ImageNet(Dataset):
#    def __init__(self, data_dir, split, transform=None):
#        assert split in ['train', 'val', 'test']
#        self.data_dir = data_dir
#        self.img_path = []
#        self.labels = []
#        self.transform = transform
#
#        if split in ['train', 'val']:
#            imagenet_split = 'train'
#        else:
#            imagenet_split = 'val'
#
#        # The class subset is taken from: https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
#        with open(os.path.join(self.data_dir, 'splits', 'imagenet.txt')) as f:
#            class_names = list(map(lambda x : x.strip(), f.readlines()))
#
#        for i, class_name in enumerate(class_names):
#            folder_path = os.path.join(
#                self.data_dir, 'imagenet', 'raw', imagenet_split, class_name)
#
#            file_names = os.listdir(folder_path)
#
#            if split is not 'test':
#                num_train = int(len(file_names) * 0.8)
#
#            for j, fid in enumerate(file_names):
#                if split == 'train' and j >= num_train: # ensures only the first 80% of data is used for training
#                    break
#                elif split == 'val' and j < num_train: # skips the first 80% of data used for training
#                    continue
#                self.img_path.append(os.path.join(folder_path, fid))
#                self.labels.append(i)
#
#        print(f"Dataset Size: {len(self.labels)}")
#        self.targets = self.labels # Sampler needs to use targets
#
#    def __len__(self):
#        return len(self.labels)
#
#    def __getitem__(self, index):
#        path = self.img_path[index]
#        label = self.labels[index]
#
#        with open(path, 'rb') as f:
#            sample = Image.open(f).convert('RGB')
#
#        if self.transform is not None:
#            sample = self.transform(sample)
#
#        return sample, label

def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


class ImageNet(data.Dataset):
    def __init__(self, data_dir, split, transform=None, target_transform=None):
        self.db_path = osp.join(data_dir, 'imagenet_train.lmdb' if split == 'train' else 'imagenet_val.lmdb')
        self.env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        #im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

def build_transform(split, input_size=224):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # train transform
    if split == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            interpolation='bicubic',
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        T.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(T.CenterCrop(input_size))

    t.append(T.ToTensor())
    t.append(T.Normalize(mean, std))
    return T.Compose(t)


