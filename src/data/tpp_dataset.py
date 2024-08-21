# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, Subset, DataLoader

from lightning import LightningModule
from src import constants

logger = logging.getLogger(__name__)

class TPPDataModule(LightningModule):
    def __init__(self, datasets, data_dir, batch_size, num_workers,
                 pin_memory=False, **kwargs):
        super().__init__()
        self.dataset = datasets['dataset']
        self.num_classes = datasets['num_classes']
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = kwargs

    def prepare_data(self):
        pass

    def setup(self, stage):
        train_dataset = TPPDataset(
            self.data_dir, self.dataset, self.num_classes, mode='train', **self.kwargs)
        self.val_dataset = TPPDataset(
            self.data_dir, self.dataset, self.num_classes, mode='val', **self.kwargs)
        self.test_dataset = TPPDataset(
            self.data_dir, self.dataset, self.num_classes, mode='test', **self.kwargs)

        # add aux_infer_dataset and dataloader for adaflood
        train_num = len(train_dataset)
        train_indices = np.arange(train_num)

        exclusion_indices = np.array([])
        inclusion_indices = np.arange(0, train_num, 1)

        aux_idx = self.kwargs.get('aux_idx', -1) # -1 for no aux, others for aux idx
        if aux_idx >= 0:
            aux_num = self.kwargs['aux_num']
            assert aux_idx < aux_num
            start_idx = int(train_num * aux_idx / float(aux_num))
            end_idx = int(train_num * (aux_idx + 1) / float(aux_num))
            exclusion_indices = np.arange(start_idx, end_idx, 1)
            inclusion_indices = np.array(list(
                set(np.arange(train_num).tolist()) - set(exclusion_indices.tolist())))

        self.train_dataset = TPPTrainWrapper(
            train_dataset, train_indices, inclusion_indices)

        self.aux_infer_dataset = TPPTrainWrapper(
            train_dataset, train_indices, exclusion_indices)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=int(self.batch_size/4), num_workers=self.num_workers,
            shuffle=False, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=int(self.batch_size/4), num_workers=self.num_workers,
            shuffle=False, pin_memory=self.pin_memory)

    def axu_infer_dataloader(self):
        return DataLoader(
            self.aux_infer_dataset, batch_size=int(self.batch_size/4), num_workers=self.num_workers,
            shuffle=False, pin_memory=self.pin_memory)



class TPPDataset(Dataset):
    real_data = ['uber_drop']

    def __init__(self, data_dir, dataset, num_classes, mode, **kwargs):
        '''
        data_dir: the root directory where all .npz files are. Default is /shared-data/TPP
        dataset: the name of a dataset
        mode: dataset type - [train, val, test]
        '''
        super(TPPDataset).__init__()
        self.mode = mode

        if dataset in self.real_data:
            data_path = os.path.join(data_dir, 'real', dataset + '.npz')
        else:
            logger.error(f'{dataset} is not valid for dataset argument'); exit()

        use_marks = kwargs.get('use_mark', True)
        data_dict = dict(np.load(data_path, allow_pickle=True))
        times = data_dict[constants.TIMES]
        marks = data_dict.get(constants.MARKS, np.ones_like(times))
        masks = data_dict.get(constants.MASKS, np.ones_like(times))
        if not use_marks:
            marks = np.ones_like(times)
        self._num_classes = num_classes

        (train_size, val_size) = (
            kwargs.get('train_size', 0.6), kwargs.get('val_size', 0.2))
        
        train_rate = kwargs.get('train_rate', 1.0)
        eval_rate = kwargs.get('eval_rate', 1.0)
        num_data = len(times)
        (start_idx, end_idx) = self._get_split_indices(
            num_data, mode=mode, train_size=train_size, val_size=val_size,
            train_rate=train_rate, eval_rate=eval_rate)

        self._times = torch.tensor(
            times[start_idx:end_idx], dtype=torch.float32).unsqueeze(-1)
        self._marks = torch.tensor(
            marks[start_idx:end_idx], dtype=torch.long).unsqueeze(-1)
        self._masks = torch.tensor(
            masks[start_idx:end_idx], dtype=torch.float32).unsqueeze(-1)

    def _sanity_check(self, time, mask):
        valid_time = time[mask.bool()]
        prev_time = valid_time[0]
        for i in range(1, valid_time.shape[0]):
            curr_time = valid_time[i]
            if curr_time < prev_time:
                logger.error(f'sanity check failed - prev time: {prev_time}, curr time: {curr_time}'); exit()
        logger.info('sanity check passed')

    def _get_split_indices(self, num_data, mode, train_size=0.6, val_size=0.2,
                           train_rate=1.0, eval_rate=1.0):
        if mode == 'train':
            start_idx = 0
            if train_size > 1.0:
                end_idx = int(train_size * train_rate)
            else:
                end_idx = int(num_data * train_size * train_rate)
        elif mode == 'val':
            if val_size > 1.0:
                start_idx = train_size
                end_idx = train_size + val_size
            else:
                start_idx = int(num_data * train_size)
                end_idx = start_idx + int(num_data * val_size * eval_rate)
        elif mode == 'test':
            if train_size > 1.0 and val_size > 1.0:
                start_idx = train_size + val_size
            else:
                start_idx = int(num_data * train_size) + int(num_data * val_size)
            end_idx = start_idx + int((num_data - start_idx) * eval_rate)
        else:
            logger.error(f'Wrong mode {mode} for dataset'); exit()
        return (start_idx, end_idx)

    def __getitem__(self, idx):
        time, mark, mask = self._times[idx], self._marks[idx], self._masks[idx]

        missing_mask = []
        input_dict = {
            constants.TIMES: time,
            constants.MARKS: mark,
            constants.MASKS: mask,
            constants.MISSING_MASKS: missing_mask,
        }
        return input_dict

    def __len__(self):
        return self._times.shape[0]

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_seq(self):
        return self._times.shape[1]

class TPPTrainWrapper(Dataset):
    def __init__(self, orig_dataset, orig_indices, inclusion_indices):
        super().__init__()

        self.orig_dataset = orig_dataset

        if len(inclusion_indices) > 0:
            self.subset_inclusion_indices = orig_indices[inclusion_indices]
        else:
            self.subset_inclusion_indices = np.array([])

        self.dataset = Subset(
            orig_dataset, self.subset_inclusion_indices)
        self.idx_to_orig_idx = {
            idx: orig_idx for idx, orig_idx in enumerate(self.subset_inclusion_indices)}

    def __getitem__(self, idx):
        input_dict = self.dataset[idx]
        input_dict[constants.INDICES] = self.idx_to_orig_idx[idx]
        return input_dict

    def __len__(self):
        return len(self.dataset)


