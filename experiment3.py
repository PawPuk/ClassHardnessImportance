import argparse
import os
import random
from typing import List, Union

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import get_config
from data import AugmentedSubset, IndexedDataset, load_dataset
from data_pruning import DataResampling
from removing_noise import NoiseRemover
from train_ensemble import ModelTrainer
from utils import set_reproducibility, load_aum_results, load_forgetting_results


class Experiment3:
    def __init__(self, dataset_name, oversampling_strategy, undersampling_strategy, remove_noise, hardness_estimator):
        self.dataset_name = dataset_name
        self.oversampling_strategy = oversampling_strategy
        self.undersampling_strategy = undersampling_strategy
        self.remove_noise = 'clean' if remove_noise else 'unclean'
        self.hardness_estimator = hardness_estimator

        self.results_file = os.path.join('Results', f"{self.remove_noise}{self.dataset_name}", 'AUM.pkl')
        self.config = get_config(dataset_name)
        self.num_classes = self.config['num_classes']
        self.num_epochs = get_config(args.dataset_name)['num_epochs']
        self.num_samples = sum(get_config(args.dataset_name)['num_training_samples'])

        self.hardness_save_dir = f"Results/{self.remove_noise}{args.dataset_name}/"


    def load_untransferred_dataset(self, train=True) -> Union[AugmentedSubset, IndexedDataset]:
        """
        Load the dataset based on dataset_name. Apply data augmentation only for training.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.config['mean'], self.config['std']),
        ])

        if self.dataset_name == 'CIFAR10':
            dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        elif self.dataset_name == 'CIFAR100':
            dataset = datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
        elif self.dataset_name == 'SVHN':
            split = 'train' if train else 'test'
            dataset = datasets.SVHN(root='./data', split=split, download=True, transform=transform)
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")

        dataset = IndexedDataset(dataset)
        if self.remove_noise == 'clean' and train:
            retained_indices = NoiseRemover(self.config, self.dataset_name, dataset).clean()
            dataset = AugmentedSubset(torch.utils.data.Subset(dataset.dataset.dataset, retained_indices))
        return dataset

    def load_hardness_estimates(self) -> List[float]:
        if self.hardness_estimator == 'AUM':
            hardness_over_models = np.array(load_aum_results(self.hardness_save_dir, self.num_epochs))
        elif self.hardness_estimator == 'Forgetting':
            aum_scores = load_aum_results(self.hardness_save_dir, self.num_epochs)
            hardness_over_models = np.array(load_forgetting_results(self.hardness_save_dir, len(aum_scores[0])))
            del aum_scores
        elif self.hardness_estimator == 'EL2N':
            el2n_path = os.path.join(self.hardness_save_dir, 'el2n_scores.pkl')
            hardness_over_models = np.array(utils.load_results(el2n_path))
        else:
            raise ValueError('The chosen hardness estimator is not supported.')

        hardness_of_ensemble = np.mean(hardness_over_models, axis=0)
        return hardness_of_ensemble

    def compute_sample_allocation(self, hardness_scores, dataset):
        """
        Compute hardness-based ratios based on class-level accuracies.
        """
        hardnesses_by_class, hardness_of_classes = {class_id: [] for class_id in range(self.num_classes)}, {}

        for i, (_, label, _) in enumerate(dataset):
            hardnesses_by_class[label].append(hardness_scores[i])
        for label in range(self.num_classes):
            if self.hardness_estimator == 'AUM':
                hardness_of_classes[label] = 1 / np.mean(hardnesses_by_class[label])
            else:
                hardness_of_classes[label] = np.mean(hardnesses_by_class[label])

        ratios = {class_id: class_hardness / sum(hardness_of_classes.values())
                  for class_id, class_hardness in hardness_of_classes.items()}
        samples_per_class = {class_id: int(round(ratio * self.num_samples)) for class_id, ratio in ratios.items()}
        return hardnesses_by_class, samples_per_class

    def get_dataloader(self, dataset, shuffle=True):
        """
        Create a DataLoader with deterministic worker initialization.
        """
        def worker_init_fn(worker_id):
            np.random.seed(self.seed + worker_id)
            random.seed(self.seed + worker_id)

        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=shuffle,
                          num_workers=2, worker_init_fn=worker_init_fn)

    @staticmethod
    def perform_data_augmentation(dataset):
        # TODO: Modify the below to work for different datasets (some might require different data augmentation)
        data_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)
        ])
        return AugmentedSubset(dataset, transform=data_augmentation)

    def main(self):
        # The value of the shuffle parameter below does not matter as we don't use the loaders.
        # TODO: Modify the below to load unaugmented data (important for resampling)
        _, training_dataset, _, test_dataset = load_dataset(self.dataset_name, self.remove_noise == 'clean', True)
        hardness_scores = self.load_hardness_estimates()

        hardnesses_by_class, samples_per_class = self.compute_sample_allocation(hardness_scores, training_dataset)

        resampler = DataResampling(training_dataset, self.num_classes, self.oversampling_strategy,
                                   self.undersampling_strategy, hardnesses_by_class, self.dataset_name)
        resampled_dataset = AugmentedSubset(resampler.resample_data(samples_per_class))

        augmented_resampled_dataset = self.perform_data_augmentation(resampled_dataset)

        resampled_loader = self.get_dataloader(augmented_resampled_dataset, shuffle=True)
        test_loader = self.get_dataloader(test_dataset, shuffle=False)

        # Print final sample allocation
        print("Samples per class after resampling in training set:")
        for class_id, count in samples_per_class.items():
            print(f"  Class {class_id}: {count}")

        model_save_dir = f"over_{self.oversampling_strategy}_under_{self.undersampling_strategy}_size_hardness"
        trainer = ModelTrainer(resampled_loader, test_loader, self.dataset_name, model_save_dir, False,
                               hardness='objective', clean_data=self.remove_noise)
        trainer.train_ensemble()


if __name__ == "__main__":
    set_reproducibility()

    parser = argparse.ArgumentParser(description="Experiment3 with Data Resampling.")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="Name of the dataset (e.g., CIFAR10, CIFAR100, SVHN)")
    parser.add_argument('--oversampling', type=str, required=True, choices=['random', 'easy', 'hard', 'SMOTE'],
                        help='Strategy used for oversampling (have to choose between `random`, `easy`, `hard`, and '
                             '`SMOTE`)')
    parser.add_argument('--undersampling', type=str, required=True, choices=['random', 'easy', 'hard', 'extreme'],
                        help='Strategy used for undersampling (have to choose between `random`, `prune_easy`, '
                             '`prune_hard`, and `prune_extreme`)')
    parser.add_argument('--remove_noise', action='store_true', help='Raise this flag to remove noise from the data.')
    parser.add_argument('hardness_estimator', type=str, choices=['EL2N', 'AUM', 'Forgetting'], default='AUM',
                        help='Specifies which instance level hardness estimator to use.')
    args = parser.parse_args()

    experiment = Experiment3(**vars(args))
    experiment.main()
