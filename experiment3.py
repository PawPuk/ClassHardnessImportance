import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from config import get_config
from data import AugmentedSubset, load_dataset
from data_pruning import DataResampling
from train_ensemble import ModelTrainer
from utils import set_reproducibility, load_aum_results, load_forgetting_results, load_results


class Experiment3:
    def __init__(self, dataset_name, oversampling, undersampling, hardness_estimator, remove_noise):
        self.dataset_name = dataset_name
        self.oversampling_strategy = oversampling
        self.undersampling_strategy = undersampling
        self.data_cleanliness = 'clean' if remove_noise else 'unclean'
        self.hardness_estimator = hardness_estimator

        self.config = get_config(dataset_name)
        self.num_classes = self.config['num_classes']
        self.num_epochs = self.config['num_epochs']
        self.num_samples = sum(self.config['num_training_samples'])

        self.hardness_save_dir = f"/mnt/parscratch/users/acq21pp/ClassHardnessImportance/Results/" \
                                 f"{self.data_cleanliness}{self.dataset_name}/"
        self.figure_save_dir = f"Figures/{self.dataset_name}/"

    def load_hardness_estimates(self):
        if self.hardness_estimator == 'AUM':
            hardness_over_models = np.array(load_aum_results(self.data_cleanliness, self.dataset_name, self.num_epochs))
        elif self.hardness_estimator == 'Forgetting':
            aum_scores = load_aum_results(self.data_cleanliness, self.dataset_name, self.num_epochs)
            hardness_over_models = np.array(
                load_forgetting_results(self.data_cleanliness, self.dataset_name, len(aum_scores[0])))
            del aum_scores
        elif self.hardness_estimator == 'EL2N':
            el2n_path = os.path.join(self.hardness_save_dir, 'el2n_scores.pkl')
            hardness_over_models = np.array(load_results(el2n_path))
        else:
            raise ValueError('The chosen hardness estimator is not supported.')

        hardness_of_ensemble = np.mean(hardness_over_models[:self.config['robust_ensemble_size']], axis=0)
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
            np.random.seed(42 + worker_id)
            random.seed(42 + worker_id)

        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=shuffle,
                          num_workers=2, worker_init_fn=worker_init_fn)

    def visualize_resampling_results(self, dataset, samples_per_class):
        class_counts = np.zeros(self.config['num_classes'], dtype=int)
        for _, label, _ in dataset:
            class_counts[label] += 1

        x = np.arange(self.config['num_classes'])
        avg_count = np.mean(class_counts)
        number_of_easy_classes = sum([samples_per_class[i] <= avg_count for i in samples_per_class.keys()])
        print(number_of_easy_classes)
        colors = ['red' if count > avg_count else 'green' for count in class_counts]

        plt.figure(figsize=(8, 5))
        plt.bar(x, class_counts, color=colors)
        plt.axhline(y=np.mean(class_counts), color='blue', linestyle='--', linewidth=2)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution (Natural Order)')
        plt.savefig(os.path.join(self.figure_save_dir, 'resampled_dataset.pdf'))

        # Plot class distribution in sorted order
        sorted_indices = np.argsort(class_counts)
        sorted_percentages = class_counts[sorted_indices]
        colors = ['red' if count > avg_count else 'green' for count in sorted_percentages]

        plt.figure(figsize=(8, 5))
        plt.bar(range(len(sorted_percentages)), sorted_percentages, color=colors)
        plt.axhline(y=np.mean(class_counts), color='blue', linestyle='--', linewidth=2)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution (Sorted Order)')
        plt.savefig(os.path.join(self.figure_save_dir, 'sorted_resampled_dataset.pdf'))

    def perform_data_augmentation(self, dataset):
        if self.dataset_name in ['CIFAR100', 'CIFAR10']:
            data_augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4)
            ])
        else:
            raise ValueError('Unsupported dataset.')
        return AugmentedSubset(dataset, transform=data_augmentation)

    def main(self):
        # The value of the shuffle parameter below does not matter as we don't use the loaders.
        _, training_dataset, _, test_dataset = load_dataset(self.dataset_name, self.data_cleanliness == 'clean', True,
                                                            False)
        hardness_scores = self.load_hardness_estimates()

        hardnesses_by_class, samples_per_class = self.compute_sample_allocation(hardness_scores, training_dataset)
        with open(os.path.join(self.hardness_save_dir, 'samples_per_class.pkl'), 'wb') as file:
            pickle.dump(samples_per_class, file)

        resampler = DataResampling(training_dataset, self.num_classes, self.oversampling_strategy,
                                   self.undersampling_strategy, hardnesses_by_class, self.dataset_name,
                                   self.hardness_estimator != 'AUM')
        resampled_dataset = AugmentedSubset(resampler.resample_data(samples_per_class))
        self.visualize_resampling_results(resampled_dataset, samples_per_class)

        augmented_resampled_dataset = self.perform_data_augmentation(resampled_dataset)

        resampled_loader = self.get_dataloader(augmented_resampled_dataset, shuffle=True)
        test_loader = self.get_dataloader(test_dataset, shuffle=False)

        print("Samples per class after resampling in training set:")
        for class_id, count in samples_per_class.items():
            print(f"  Class {class_id}: {count}")

        model_save_dir = f"over_{self.oversampling_strategy}_under_{self.undersampling_strategy}_size_hardness"
        trainer = ModelTrainer(len(resampled_dataset), resampled_loader, test_loader, self.dataset_name, model_save_dir,
                               False, clean_data=self.data_cleanliness == 'clean')
        trainer.train_ensemble()


if __name__ == "__main__":
    set_reproducibility()

    parser = argparse.ArgumentParser(description="Experiment3 with Data Resampling.")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="Name of the dataset (e.g., CIFAR10, CIFAR100, SVHN)")
    parser.add_argument('--oversampling', type=str, required=True, choices=['random', 'easy', 'hard', 'SMOTE', 'none'],
                        help='Strategy used for oversampling (have to choose between `random`, `easy`, `hard`, '
                             '`SMOTE`, and `none`)')
    parser.add_argument('--undersampling', type=str, required=True, choices=['easy', 'none'],
                        help='Strategy used for undersampling (have to choose between `random`, `prune_easy`, '
                             '`prune_hard`, `prune_extreme`, and `none`)')
    parser.add_argument('--hardness_estimator', type=str, choices=['EL2N', 'AUM', 'Forgetting'], default='AUM',
                        help='Specifies which instance level hardness estimator to use.')
    parser.add_argument('--remove_noise', action='store_true', help='Raise this flag to remove noise from the data.')
    args = parser.parse_args()

    experiment = Experiment3(**vars(args))
    experiment.main()
