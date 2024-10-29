import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from utils import get_config
from train_ensemble import ModelTrainer


# Function to get data transforms
def get_data_transforms(dataset_name):
    if dataset_name == 'CIFAR10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset_name == 'CIFAR100':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported. Choose either CIFAR10 or CIFAR100.")

    return train_transform, test_transform


# Function to load dataset
def get_dataloader(dataset_name, batch_size, train_transform, test_transform):
    if dataset_name == 'CIFAR10':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    elif dataset_name == 'CIFAR100':
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported. Choose either CIFAR10 or CIFAR100.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


# Main function
def main(dataset_name):
    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Get the dataset transforms based on the dataset_name
    train_transform, test_transform = get_data_transforms(dataset_name)

    # Load the dataset
    batch_size = get_config(dataset_name)['batch_size']
    training_loader, test_loader = get_dataloader(dataset_name, batch_size, train_transform, test_transform)

    # Create an instance of ModelTrainer
    trainer = ModelTrainer(training_loader, test_loader, dataset_name)

    # Train the ensemble of models
    trainer.train_ensemble()


if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-10 or CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True, choices=['CIFAR10', 'CIFAR100'],
                        help='Dataset name: CIFAR10 or CIFAR100')

    # Parse arguments
    args = parser.parse_args()

    # Run main function
    main(args.dataset_name)
