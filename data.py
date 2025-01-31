import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from config import get_config
from removing_noise import NoiseRemover


SEED = 42


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        # To make the training faster we transform the dataset into a TensorDataset
        if not isinstance(dataset, TensorDataset):
            data_list, label_list = [], []
            for i in range(len(dataset)):
                data, label = dataset[i]
                data_list.append(data.unsqueeze(0))  # Add a dimension for torch.cat (this is for number of samples)
                label_list.append(torch.tensor(label)) # Necessary because some datasets return labels as integers
            data_tensor = torch.cat(data_list, dim=0)
            label_tensor = torch.tensor(label_list)
            dataset = TensorDataset(data_tensor, label_tensor)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        return data, label, idx


class AugmentedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Get the original data and label from the subset
        data, label, _ = self.subset[idx]

        # Apply the transformations to the data
        if self.transform:
            data = self.transform(data)

        return data, label, idx


def load_dataset(dataset_name, remove_noise, shuffle):
    config = get_config(dataset_name)

    # TODO: we might want to make the below dataset-specific and enable more datasets
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
    if dataset_name == 'CIFAR10':
        training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                    transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    elif dataset_name == 'CIFAR100':
        training_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                     transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    training_set = IndexedDataset(training_set)
    test_set = IndexedDataset(test_set)
    if remove_noise:
        retained_indices = NoiseRemover(config, dataset_name, training_set).clean()
        training_set = AugmentedSubset(IndexedDataset(torch.utils.data.Subset(training_set.dataset, retained_indices)))

    def worker_init_fn(worker_id):
        np.random.seed(SEED + worker_id)
        random.seed(SEED + worker_id)

    training_loader = DataLoader(training_set, batch_size=config['batch_size'], shuffle=shuffle, num_workers=2,
                                 worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=2,
                             worker_init_fn=worker_init_fn)

    return training_loader, training_set, test_loader, test_set