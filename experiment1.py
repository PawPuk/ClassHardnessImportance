import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from datasets import CINIC10
from removing_noise import NoiseRemover
from train_ensemble import ModelTrainer
from utils import get_config, IndexedDataset


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
    elif dataset_name == 'SVHN':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4524, 0.4525, 0.4690),
                                 (0.2194, 0.2268, 0.2280)),
        ])
    elif dataset_name == 'CINIC10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                                 (0.24205776, 0.23828046, 0.25874835)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                                 (0.24205776, 0.23828046, 0.25874835)),
        ])
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported. Choose either CIFAR10 or CIFAR100.")

    return train_transform, test_transform


# Function to load dataset
def get_dataloader(dataset_name, batch_size, train_transform, test_transform, seed, remove_noise):
    if dataset_name == 'CIFAR10':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    elif dataset_name == 'CIFAR100':
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    elif dataset_name == 'SVHN':
        train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=train_transform)
        test_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)
    elif dataset_name == 'CINIC10':
        train_set = CINIC10(root_dir='/mnt/parscratch/users/acq21pp/CINIC-10', split='train', transform=train_transform)
        test_set = CINIC10(root_dir='/mnt/parscratch/users/acq21pp/CINIC-10', split='test', transform=test_transform)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported. Choose either CIFAR10 or CIFAR100.")

    train_set = IndexedDataset(train_set)
    test_set = IndexedDataset(test_set)
    if remove_noise:
        NoiseRemover(dataset_name, train_set).clean()

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                              worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2,
                             worker_init_fn=worker_init_fn)

    return train_loader, test_loader


# Main function
def main(dataset_name: str, remove_noise: bool):
    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Get the dataset transforms based on the dataset_name
    train_transform, test_transform = get_data_transforms(dataset_name)

    # Load the dataset
    batch_size = get_config(dataset_name)['batch_size']
    training_loader, test_loader = get_dataloader(dataset_name, batch_size, train_transform, test_transform, seed,
                                                  remove_noise)

    # Create an instance of ModelTrainer
    clean_data = 'clean' if remove_noise else 'unclean'
    trainer = ModelTrainer(training_loader, test_loader, dataset_name, compute_aum=True, clean_data=clean_data)

    # Train the ensemble of models
    trainer.train_ensemble()


if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-10 or CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR10', 'CIFAR100', 'SVHN', 'CINIC10'], help='Dataset name: CIFAR10 or CIFAR100')
    parser.add_argument('--remove_noise', action='store_true',
                        help='Raise this flag to remove noise from the data.')

    # Parse arguments
    args = parser.parse_args()

    # Run main function
    main(args.dataset_name, args.remove_noise)
