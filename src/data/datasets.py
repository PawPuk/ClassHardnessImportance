"""Wrappers for datasets that allowing mapping between"""

from typing import Tuple

import torch
from torch.utils.data import TensorDataset


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, is_test=False, transform=None):
        # To improve speed, we transform the dataset into a TensorDataset (only viable if no augmentation is applied)
        if not isinstance(dataset, TensorDataset) and is_test:
            data_list, label_list = [], []
            for i in range(len(dataset)):
                data, label = dataset[i]
                data_list.append(data)
                label_list.append(torch.tensor(label))  # Necessary because some datasets return labels as integers
            data_tensor = torch.stack(data_list)
            label_tensor = torch.tensor(label_list)
            dataset = TensorDataset(data_tensor, label_tensor)

        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        data, label = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, label, idx

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class AugmentedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Get the original data and label from the subset
        data, label, _ = self.subset[idx]

        # Apply the transformations to the data
        if self.transform:
            data = self.transform(data)
        return data, label, idx

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
