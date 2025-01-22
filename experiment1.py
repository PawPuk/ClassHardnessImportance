import argparse
import random

import numpy as np
import torch
from train_ensemble import ModelTrainer
from utils import load_dataset


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

    training_loader, _, test_loader, _ = load_dataset(dataset_name, remove_noise, seed, True)

    # Create an instance of ModelTrainer
    trainer = ModelTrainer(training_loader, test_loader, dataset_name, estimate_hardness=True, clean_data=remove_noise)

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
