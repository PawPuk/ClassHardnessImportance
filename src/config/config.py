"""
Configuration module.

This module defines global experiment parameters used across the repository.

Experimental design
-------------------

The experiments are designed to measure whether hardness-based resampling
reduces performance disparities across classes.

To ensure statistical robustness, we account for two sources of randomness:

1. Model initialization randomness
2. Dataset generation randomness (pruning or resampling)

To control for these factors, experiments are performed using a two-level design:

    number of dataset variants  ×  number of models per dataset

For example:

    4 datasets × 4 models per dataset = 16 trained models

Each dataset variant is generated using different random seeds during
pruning or resampling (see utils/reproducibility.py for seed generation).
For each dataset variant an ensemble of models is trained, allowing us
to compute stable performance estimates.

Important parameters
--------------------

num_datasets
    Number of independently generated dataset variants.

num_models_per_dataset
    Number of models trained for each dataset variant.

num_models_for_hardness
    Number of models used to estimate sample hardness during
    baseline ensemble training.

Runtime considerations
---------------------

Larger configurations increase statistical robustness but significantly
increase training time.

Typical settings:

    4×4  – used in the paper experiments
    2×2  – recommended for faster experimentation
    1×1  – quick but non-robust experimentation
"""

import os
from typing import Dict, List, Tuple, Union

import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = '.'

dataset_configs = {
    'CIFAR10': {
        # Training hyperparameters
        'batch_size': 128,
        'num_epochs': 200,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_decay_milestones': [60, 120, 160],

        # Dataset information
        'num_classes': 10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
        'num_training_samples': [5000 for _ in range(10)],
        'num_test_samples': [1000 for _ in range(10)],
        'class_names': ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],

        # Experimental robustness parameters
        'num_datasets': 4,              # number of dataset variants
        'num_models_per_dataset': 4,    # number of models trained on each dataset variant

        # Hardness estimation
        'num_models_for_hardness': 10,   # ensemble size used to compute hardness in train_baseline_models.py

        # Other
        'save_epoch': 20,
        'save_dir': os.path.join(ROOT, 'Models/')
    },
    'CIFAR100': {
        # Training hyperparameters
        'batch_size': 128,
        'num_epochs': 200,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_decay_milestones': [60, 120, 160],

        # Dataset information
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
            'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
            'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
            'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'],


        # Experimental robustness parameters
        'num_datasets': 4,              # number of dataset variants
        'num_models_per_dataset': 4,    # number of models trained on each dataset variant

        # Hardness estimation
        'num_models_for_hardness': 10,  # ensemble size used to compute hardness in train_baseline_models.py

        # Other
        'save_epoch': 20,
        'save_dir': os.path.join(ROOT, 'Models/')
    }
}


def get_config(dataset_name: str) -> Dict[str, Union[int, float, str, List[int], List[float], List[str], Tuple[
                                     float, float, float]]]:
    if dataset_name in dataset_configs:
        config = dataset_configs[dataset_name]
        config['probe_base_seed'] = 42
        config['probe_seed_step'] = 42
        config['probe_dataset_step'] = 420_000
        return config
    else:
        raise ValueError(f"Configuration for dataset {dataset_name} not found!")
