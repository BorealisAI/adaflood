# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import os
import copy
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from src import constants
from src.utils.utils import collate_fn, generate_noisy_labels


class CLSDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(self, datasets, data_dir, batch_size, num_workers,
                 pin_memory, **kwargs):
        super().__init__()
        data_config = copy.deepcopy(datasets)
        self.dataset = data_config.pop('dataset')
        self.num_classes = data_config.pop('num_classes')
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_config = data_config
        self.kwargs = kwargs

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        if self.dataset == 'cifar100':
            datasets.CIFAR100(self.data_dir, train=True, download=True)
            datasets.CIFAR100(self.data_dir, train=False, download=True)
        else:
            raise NotImplementedError(f'dataset: {self.dataset} is not implemented')


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        train_rate = self.kwargs.get('train_rate', 1.0)
        alpha = self.kwargs.get('alpha', 0.0)
        split_path = os.path.join(
                self.data_dir, 'splits', f'{self.dataset}_train_val_rate{train_rate}.npz')
        noisy_label_path = os.path.join(
                self.data_dir, 'splits', f'{self.dataset}_noisy_labels_alpha{alpha}.npy')
        noisy_indices_path = os.path.join(
                self.data_dir, 'splits', f'{self.dataset}_noisy_indices_alpha{alpha}.npy')

        if self.dataset == 'cifar100':
            # data transformations
            self.train_transforms = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomRotation(15),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5070, 0.4895, 0.4409), (0.2673, 0.2564, 0.2761))])

            self.test_transforms = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5070, 0.4895, 0.4409), (0.2673, 0.2564, 0.2761))])

            # default train: 40_000, val: 10_000
            if os.path.exists(split_path):
                shuffled_indices = np.load(split_path)
                train_indices = shuffled_indices['train']
                val_indices = shuffled_indices['val']
                train_num = len(train_indices)
            else:
                shuffled_indices = np.arange(50_000)
                np.random.shuffle(shuffled_indices)
                train_num = int(40_000 * train_rate)
                train_indices = shuffled_indices[:train_num]
                val_indices = shuffled_indices[40_000:]
                np.savez(split_path, train=train_indices, val=val_indices)

            trainset = datasets.CIFAR100(
                self.data_dir, train=True, transform=self.train_transforms)
            self.data_test = datasets.CIFAR100(
                self.data_dir, train=False, transform=self.test_transforms)
            self.data_val = Subset(trainset, val_indices)
        else:
            raise NotImplementedError(f'dataset: {self.dataset} is not implemented')

        # imbalanaced setting
        imb_factor = self.kwargs.get('imb_factor', 1.0)
        if imb_factor < 1.0:
            prev_train_num = train_num
            train_labels = np.array(trainset.targets)[train_indices]
            img_num_per_class = self._get_image_num_per_class(
                imb_factor, train_labels)
            train_indices = self._gen_imbalanced_data(
                img_num_per_class, trainset, train_indices)
            train_num = len(train_indices)
            print(f'updated the number of training set: {prev_train_num} -> {train_num} for imb factor={imb_factor}')

        # further split the trainset for adaflood
        exclusion_indices = np.array([])
        inclusion_indices = np.arange(0, train_num, 1)
        aux_idx = self.kwargs.get('aux_idx', -1) # -1 for no aux, others for aux idx
        aux_num = self.kwargs['aux_num']
        if aux_idx >= 0 and aux_num > 0:
            assert aux_idx < aux_num
            start_idx = int(train_num * aux_idx / float(aux_num))
            end_idx = int(train_num * (aux_idx + 1) / float(aux_num))
            exclusion_indices = np.arange(start_idx, end_idx, 1)
            inclusion_indices = np.array(list(
                set(np.arange(train_num).tolist()) - set(exclusion_indices.tolist())))

        print(f'Num inclusion: {len(inclusion_indices)}/{train_num}, num exlcusion: {len(exclusion_indices)}/{train_num}')

        # add noisy label setting
        noisy_idx_to_label = {}
        if alpha > 0.0:
            if os.path.exists(noisy_label_path):
                noisy_idx_to_label = np.load(noisy_label_path, allow_pickle=True).item()
            else:
                num_noisy_samples = int(alpha * train_num)
                noisy_indices = np.random.choice(train_indices, size=num_noisy_samples, replace=False)
                labels = np.array([trainset[idx][1] for idx in noisy_indices])
                noisy_labels = generate_noisy_labels(labels, self.num_classes)

                noisy_idx_to_label = {
                    idx: label for idx, label in zip(noisy_indices, noisy_labels)}

                np.save(noisy_label_path, noisy_idx_to_label)

        print(f'Num noisy labels: {len(noisy_idx_to_label.keys())}/{train_num}')

        if alpha < 0.0:
            if os.path.exists(noisy_indices_path):
                noisy_indices = np.load(noisy_indices_path, allow_pickle=True).item()
            else:
                num_noisy_samples = int(abs(alpha) * train_num)
                noisy_indices = np.random.choice(
                    train_indices, size=num_noisy_samples, replace=False)

                np.save(noisy_indices_path, noisy_indices)
            noisy_idx_to_label = noisy_indices


        self.data_train = CLSTrainWrapper(
            trainset, train_indices, inclusion_indices, noisy_idx_to_label, self.dataset)

        self.data_aux_infer = CLSTrainWrapper(
            trainset, train_indices, exclusion_indices, noisy_idx_to_label, self.dataset)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def axu_infer_dataloader(self):
        return DataLoader(
            dataset=self.data_aux_infer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )

    def indexed_dataloader(self, indices):
        subset_sampler = SubsetRandomSampler(indices)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=subset_sampler
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def _get_image_num_per_class(
        self, imb_factor, labels, imb_type="exp"):
        img_num_per_cls = []

        if imb_type == "exp":
            for cls_idx in range(self.num_classes):
                img_max = np.sum(labels == cls_idx)
                img_num = int(img_max * (
                    imb_factor ** (cls_idx / (self.num_classes - 1.0))))
                img_num_per_cls.append(img_num)
        else:
            raise NotImplementedError(f'Imbalance type: {imb_type} is not implemented')

        return img_num_per_cls

    def _gen_imbalanced_data(self, image_num_per_class, dataset, valid_indices):
        labels = np.array(dataset.targets, dtype=np.int64)
        classes = np.unique(labels)

        imb_validity_bool = [False] * len(valid_indices)
        count_per_class_dict = {
            cls: img_num for cls, img_num in zip(classes, image_num_per_class)}
        cls_index_dict = {cls: np.where(labels == cls)[0] for cls in classes}
        for i in range(len(valid_indices)):
            for cls in classes:
                if valid_indices[i] not in cls_index_dict[cls]:
                    continue
                else:
                    img_num = count_per_class_dict[cls]
                    if img_num > 0:
                        imb_validity_bool[i] = True
                        count_per_class_dict[cls] -= 1
                    break

        imb_validity_bool = np.array(imb_validity_bool)
        assert len(imb_validity_bool) == len(valid_indices)
        assert np.sum(image_num_per_class) == np.sum(imb_validity_bool)

        imb_train_indices = valid_indices[imb_validity_bool]
        return imb_train_indices

class CLSTrainWrapper(Dataset):
    def __init__(self, orig_dataset, orig_indices, inclusion_indices, noisy_idx_to_label, dataset_name):
        super().__init__()

        self.dataset_name = dataset_name
        self.orig_dataset = orig_dataset
        self.noisy_idx_to_label = noisy_idx_to_label

        if len(inclusion_indices) > 0:
            self.subset_inclusion_indices = orig_indices[inclusion_indices]
        else:
            self.subset_inclusion_indices = np.array([])

        self.dataset = Subset(
            orig_dataset, self.subset_inclusion_indices)
        self.idx_to_orig_idx = {
            idx: orig_idx for idx, orig_idx in enumerate(self.subset_inclusion_indices)}

        # Define the mean and standard deviation of the Gaussian noise
        self.noise_scale = 10.0

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        orig_idx = self.idx_to_orig_idx[idx]

        if orig_idx in self.noisy_idx_to_label:
            if isinstance(self.noisy_idx_to_label, dict):
                label = self.noisy_idx_to_label[orig_idx]
            else:
                noise = torch.randn_like(img) * self.noise_scale
                img = img + noise

        input_dict = {
            constants.IMAGES: img,
            constants.LABELS: label,
            constants.INDICES: orig_idx
        }
        return input_dict

    def __len__(self):
        return len(self.dataset)
