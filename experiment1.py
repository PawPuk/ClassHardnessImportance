import argparse

from data import load_dataset
from train_ensemble import ModelTrainer
from utils import set_reproducibility


def main(dataset_name: str, remove_noise: bool):
    training_loader, training_set, test_loader, _ = load_dataset(dataset_name, remove_noise, True, True)
    training_set_size = len(training_set)

    trainer = ModelTrainer(training_set_size, training_loader, test_loader, dataset_name, estimate_hardness=True,
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
