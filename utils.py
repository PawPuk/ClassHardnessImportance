import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from removing_noise import NoiseRemover


dataset_configs = {
    'CIFAR10': {
        'batch_size': 128,
        'num_epochs': 200,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_decay_milestones': [60, 120, 160],
        'save_epoch': 20,
        'save_dir': './Models/',
        'timings_dir': './Timings/',
        'num_models': 10,
        'num_classes': 10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
        'num_training_samples': [5000 for _ in range(10)],
        'num_test_samples': [1000 for _ in range(10)],
        'safe_pruning_ratios': [48.8, 74.22, 29.08, 10.64, 45.8, 31.44, 54.92, 58.36, 67.28, 63.66]
    },
    'CIFAR100': {
        'batch_size': 128,
        'num_epochs': 200,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_decay_milestones': [60, 120, 160],
        'save_epoch': 20,
        'save_dir': './Models/',
        'timings_dir': './Timings/',
        'num_models': 10,
        'num_classes': 100,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
        'num_training_samples': [500 for _ in range(100)],
        'num_test_samples': [100 for _ in range(100)],
        'class_names': [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
            'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
            'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
            'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
            'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    },
    'SVHN': {
        'batch_size': 128,
        'num_epochs': 200,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_decay_milestones': [60, 120, 160],
        'save_epoch': 20,
        'save_dir': './Models/',
        'timings_dir': './Timings/',
        'num_models': 10,
        'num_classes': 10,
        'mean': (0.4377, 0.4438, 0.4728),
        'std': (0.1980, 0.2010, 0.1970),
        'num_training_samples': [4948, 13861, 10585, 8497, 7458, 6882, 5727, 5595, 5045, 4659]
    },
    'CINIC10': {
        'batch_size': 128,
        'num_epochs': 200,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_decay_milestones': [60, 120, 160],
        'save_epoch': 20,
        'save_dir': './Models/',
        'timings_dir': './Timings/',
        'num_models': 20,
        'num_classes': 10,
        'mean': (0.4789, 0.4723, 0.4305),
        'std': (0.2421, 0.2383, 0.2587),
        'num_training_samples': [9000 for _ in range(10)]
    }
}

"""Other setting for CIFAR100 proposed by ChatGPT that appears to give better results
'CIFAR100': {
        'batch_size': 128,
        'num_epochs': 200,
        'lr': 0.05,
        'momentum': 0.85,
        'weight_decay': 0.0001,
        'lr_decay_milestones': [50, 150, 225],
        'save_epoch': 25,
        'save_dir': './Models/',
        'timings_dir': './Timings/',
        'num_models': 8,
        'num_classes': 100
    }
"""


def get_config(dataset_name):
    """
    Fetch the appropriate configuration based on the dataset name.
    """
    if dataset_name in dataset_configs:
        config = dataset_configs[dataset_name]
        config['probe_base_seed'] = 42
        config['probe_seed_step'] = 42
        config['new_base_seed'] = 4242
        config['new_seed_step'] = 42
        return config
    else:
        raise ValueError(f"Configuration for dataset {dataset_name} not found!")


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
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


def load_dataset(dataset_name, remove_noise, seed, shuffle):
    config = get_config(dataset_name)

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

    print(type(training_set))
    print(type(test_set))

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    training_loader = DataLoader(training_set, batch_size=config['batch_size'], shuffle=shuffle, num_workers=2,
                                 worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=2,
                             worker_init_fn=worker_init_fn)

    return training_loader, training_set, test_loader, test_set
