import argparse
import os
import pickle
import random
from typing import Dict, Union

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import get_config

from data_pruning import DataResampling
from removing_noise import NoiseRemover
from train_ensemble import ModelTrainer
from utils import AugmentedSubset, IndexedDataset, load_dataset, set_reproducibility


class Experiment3:
    def __init__(self, dataset_name, oversampling_strategy, undersampling_strategy, remove_noise):
        self.dataset_name = dataset_name
        self.oversampling_strategy = oversampling_strategy
        self.undersampling_strategy = undersampling_strategy
        self.remove_noise = 'clean' if remove_noise else 'unclean'

        self.results_file = os.path.join('Results', f"{self.remove_noise}{self.dataset_name}", 'AUM.pkl')
        self.config = get_config(dataset_name)
        self.num_classes = self.config['num_classes']
        self.num_epochs = get_config(args.dataset_name)['num_epochs']
        self.num_samples = sum(get_config(args.dataset_name)['num_training_samples'])

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

    def load_results(self):
        """
        Load the computed accuracies from results_file.
        """
        with open(self.results_file, 'rb') as file:
            return pickle.load(file)

    def compute_sample_allocation(self, aum_scores, dataset):
        """
        Compute hardness-based ratios based on class-level accuracies.
        """

        hardnesses_by_class, hardness_of_classes = {class_id: [] for class_id in range(self.num_classes)}, {}
        print(type(dataset))
        for i, (_, label, _) in enumerate(dataset):
            hardnesses_by_class[label].append(aum_scores[i])
        for label in range(self.num_classes):
            hardness_of_classes[label] = np.mean(hardnesses_by_class[label])
        inverted_ratios = {class_id: 1 / class_hardness for class_id, class_hardness in hardness_of_classes.items()}
        normalized_ratios = {class_id: inverted_ratio / sum(inverted_ratios.values())
                             for class_id, inverted_ratio in inverted_ratios.items()}
        samples_per_class = {class_id: int(round(normalized_ratio * self.num_samples))
                             for class_id, normalized_ratio in normalized_ratios.items()}
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
        data_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)
        ])
        return AugmentedSubset(dataset, transform=data_augmentation)

    def main(self):
        # The value of the shuffle parameter below does not matter as we don't use the loaders.
        _, training_dataset, _, test_dataset = load_dataset(self.dataset_name, self.remove_noise == 'clean', True)
        AUM_over_epochs_and_models = self.load_results()
        for model_idx, model_list in enumerate(AUM_over_epochs_and_models):
            AUM_over_epochs_and_models[model_idx] = [sample for sample in model_list if len(sample) > 0]

        AUM_scores_over_models = [
            [
                sum(model_list[sample_idx][epoch_idx] for epoch_idx in range(self.num_epochs)) / self.num_epochs
                for sample_idx in range(len(AUM_over_epochs_and_models[0]))
            ]
            for model_list in AUM_over_epochs_and_models
        ]
        AUM_scores = np.mean(AUM_scores_over_models, axis=0)

        # Compute hardness-based ratios and sample allocation
        hardnesses_by_class, samples_per_class = self.compute_sample_allocation(AUM_scores, training_dataset)

        # Perform resampling
        resampler = DataResampling(training_dataset, self.num_classes, self.oversampling_strategy,
                                   self.undersampling_strategy, hardnesses_by_class, self.dataset_name)
        resampled_dataset = AugmentedSubset(resampler.resample_data(samples_per_class))

        augmented_resampled_dataset = self.perform_data_augmentation(resampled_dataset)

        # Get DataLoaders
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
    args = parser.parse_args()

    experiment = Experiment3(args.dataset_name, args.oversampling, args.undersampling, args.remove_noise)
    experiment.main()
