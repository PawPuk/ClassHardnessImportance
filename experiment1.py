"""
This is the first important module that allows training the ensemble on the balanced, full-sized datasets.

Important parts:
* Requires that the `num_datasets` in the dataset_configs variable from config file to be set to 1, as there is only
one version of the full dataset. Setting this variable to more than 1 makes sense only in further experiments where we
work with pruned and resampled datasets, which are obtained through non-deterministic means.
"""

import argparse

from config import get_config
from data import load_dataset
from train_ensemble import ModelTrainer
from utils import set_reproducibility


def main(dataset_name: str, remove_noise: bool):
    """Main function that runs the code."""
    if get_config(dataset_name)['num_datasets'] != 1:
        raise Exception(f"This code requires `get_config(dataset_name)['num_datasets']` to be set to 1, not "
                        f"{get_config(dataset_name)['num_datasets']}.")
    training_loader, training_set, test_loader, _ = load_dataset(dataset_name, remove_noise, True, True)
    training_set_size = len(training_set)
    training_loaders = [training_loader]

    trainer = ModelTrainer(training_set_size, training_loaders, test_loader, dataset_name, estimate_hardness=True,
                           clean_data=remove_noise)

    trainer.train_ensemble()


if __name__ == '__main__':
    set_reproducibility()

    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-10 or CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR10', 'CIFAR100'], help='Dataset name: CIFAR10 or CIFAR100')
    parser.add_argument('--remove_noise', action='store_true',
                        help='Raise this flag to remove noise from the data.')

    args = parser.parse_args()

    main(args.dataset_name, args.remove_noise)
