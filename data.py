"""The data module: Provides two core Dataset subclasses, and the method for loading the data."""

import os
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from config import get_config, ROOT
from removing_noise import NoiseRemover


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


def perform_data_augmentation(dataset: AugmentedSubset, dataset_name: str) -> AugmentedSubset:
    """Applies data augmentation to the dataset. It firstly converts the images from Tensor to PIL to ensure the whole
    process is intact. This is useful in scenarios where we initially load the training dataset without applying data
    augmentation - load_dataset() with apply_augmentation=False. This function allows us to apply the augmentation to
    the training set."""
    mean = get_config(dataset_name)['mean']
    std = get_config(dataset_name)['std']

    data_augmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
    ])
    return AugmentedSubset(dataset, transform=data_augmentation)


def get_transform(dataset_name: str, apply_augmentation: bool, config: Dict[str, Union[int, float, List[int],
                  List[float], List[str], Tuple[float, float, float]]]
                  ) -> Tuple[transforms.Compose, transforms.Compose]:
    """For getting the transformation to the training and test sets."""
    if apply_augmentation and dataset_name in ['CIFAR100', 'CIFAR10']:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std']),
        ])
    else:
        train_transform = transforms.ToTensor()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
    return train_transform, test_transform


def get_dataloader(dataset, batch_size: int, shuffle: bool = False):
    """Create a DataLoader with deterministic worker initialization."""
    def worker_init_fn(worker_id):
        """Set the seed for workers"""
        np.random.seed(42 + worker_id)
        random.seed(42 + worker_id)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, worker_init_fn=worker_init_fn)


def load_dataset(dataset_name: str, remove_noise: bool = False, shuffle: bool = False, apply_augmentation: bool = False
                 ) -> Tuple[DataLoader[Union[IndexedDataset, AugmentedSubset]], Union[
        IndexedDataset, AugmentedSubset], DataLoader[IndexedDataset], IndexedDataset]:
    """Allows to load dataset, remove label noise from it, and gives precise control over shuffling and augmentation.
    Currently only supports CIFAR10 and CIFAR100.

    :param dataset_name: Specifies name of the dataset to load (only accepts `CIFAR10` and `CIFAR100`).
    :param remove_noise: Raise this flag to remove the label noise from the training dataset using AUM.
    :param shuffle: Raise this flag to shuffle the training dataset.
    :param apply_augmentation: Raise this flag to apply data augmentation to the training set.

    :return: Tuple containing DataLoader for the training set, training set, DataLoader for the test set, and test set.
    """
    config = get_config(dataset_name)

    train_transform, test_transform = get_transform(dataset_name, apply_augmentation, config)
    if dataset_name == 'CIFAR10':
        training_set = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, 'data'), download=True,
                                                    transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, 'data'), train=False, download=True,
                                                transform=test_transform)
    elif dataset_name == 'CIFAR100':
        training_set = torchvision.datasets.CIFAR100(root=os.path.join(ROOT, 'data'), download=True,
                                                     transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root=os.path.join(ROOT, 'data'), train=False, download=True,
                                                 transform=test_transform)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    training_set = IndexedDataset(training_set, apply_augmentation is False)
    test_set = IndexedDataset(test_set, True)
    if remove_noise:
        retained_indices = NoiseRemover(config, dataset_name, training_set).clean()
        training_set = AugmentedSubset(IndexedDataset(torch.utils.data.Subset(training_set.dataset, retained_indices)))

    training_loader = get_dataloader(training_set, config['batch_size'], shuffle)
    test_loader = get_dataloader(test_set, config['batch_size'])

    return training_loader, training_set, test_loader, test_set
