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
from utils import AugmentedSubset, IndexedDataset


class Experiment3:
    def __init__(self, dataset_name, desired_dataset_size, oversampling_strategy, undersampling_strategy,
                 class_hardness_estimation, remove_noise):
        self.dataset_name = dataset_name
        self.desired_dataset_size = desired_dataset_size
        self.oversampling_strategy = oversampling_strategy
        self.undersampling_strategy = undersampling_strategy
        self.hardness_estimation = class_hardness_estimation
        self.remove_noise = 'clean' if remove_noise else 'unclean'

        self.results_file = os.path.join('Results', f"{self.remove_noise}{self.dataset_name}", 'el2n_scores.pkl')
        self.config = get_config(dataset_name)
        self.num_classes = self.config['num_classes']

        # Reproducibility settings
        self.seed = 42
        self.set_reproducibility()

    def set_reproducibility(self):
        """
        Ensure reproducibility by setting seeds and configuring PyTorch settings.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
            NoiseRemover(self.config, self.dataset_name, dataset).clean()
        return dataset

    def load_results(self):
        """
        Load the computed accuracies from results_file.
        """
        with open(self.results_file, 'rb') as file:
            return pickle.load(file)

    def estimate_hardness_using_accuracy(self, accuracies):
        """
        Compute hardness-based ratios based on class-level accuracies.
        """
        class_accumulator = {i: 0 for i in range(self.num_classes)}
        class_counts = {i: 0 for i in range(self.num_classes)}

        for model_class_acc in accuracies:
            for class_id, acc in model_class_acc.items():
                class_accumulator[class_id] += acc
                class_counts[class_id] += 1

        avg_class_accuracies = {class_id: class_accumulator[class_id] / class_counts[class_id]
                                for class_id in class_accumulator}

        # Compute ratios as 1 / accuracy
        return {class_id: 1 / acc for class_id, acc in avg_class_accuracies.items()}

    def estimate_hardness_using_el2n(self, el2n_scores):
        """
        Compute hardness-based ratios based on class-level accuracies.
        """
        class_accumulator = {i: 0 for i in range(self.num_classes)}
        class_counts = {i: 0 for i in range(self.num_classes)}

        for class_id, current_class_el2n_scores in el2n_scores.items():
            for data_sample_scores in current_class_el2n_scores:
                average_score = sum(data_sample_scores) / len(data_sample_scores)
                class_accumulator[class_id] += average_score
                class_counts[class_id] += 1

        avg_class_el2n_scores = {class_id: class_accumulator[class_id] / class_counts[class_id]
                                 for class_id in class_accumulator}

        # Compute ratios as el2n
        return {class_id: el2n for class_id, el2n in avg_class_el2n_scores.items()}

    def estimate_hardness_based_on_safe_pruning_ratios(self):
        """
        Compute hardness-based ratios based on how many easy samples can be removed from each class without impacting
        the performance on those classes.
        """
        safe_pruning_ratios = self.config['safe_pruning_ratios']
        return {class_id: 1 / safe_pruning_ratios[class_id] for class_id in range(self.num_classes)}

    def compute_hardness_based_ratios(self, class_accuracies, class_el2n_scores):
        if self.hardness_estimation == 'accuracy':
            return self.estimate_hardness_using_accuracy(class_accuracies)
        elif self.hardness_estimation == 'EL2N':
            return self.estimate_hardness_using_el2n(class_el2n_scores)
        else:
            return self.estimate_hardness_based_on_safe_pruning_ratios()

    def compute_sample_allocation(self, ratios) -> Dict[int, float]:
        """
        Compute the number of samples required for each class to match the desired_dataset_size.
        """
        total_ratio = sum(ratios.values())
        normalized_ratios = {class_id: ratio / total_ratio for class_id, ratio in ratios.items()}

        # Allocate samples based on normalized ratios
        samples_per_class = {class_id: int(round(normalized_ratio * self.desired_dataset_size))
                             for class_id, normalized_ratio in normalized_ratios.items()}

        # Adjust to ensure total matches desired_dataset_size
        total_allocated = sum(samples_per_class.values())
        if total_allocated != self.desired_dataset_size:
            difference = self.desired_dataset_size - total_allocated
            sorted_classes = sorted(samples_per_class.keys(), key=lambda cid: -ratios[cid])
            for class_id in sorted_classes:
                samples_per_class[class_id] += 1 if difference > 0 else -1
                difference += -1 if difference > 0 else 1
                if difference == 0:
                    break

        return samples_per_class

    def resample_dataset(self, dataset, all_el2n_scores, class_el2n_scores, samples_per_class) -> AugmentedSubset:
        """
        Use DataResampling to modify the dataset to match the samples_per_class.
        """
        resampler = DataResampling(dataset, self.num_classes, self.oversampling_strategy, self.undersampling_strategy,
                                   all_el2n_scores, class_el2n_scores, self.dataset_name)
        return AugmentedSubset(resampler.resample_data(samples_per_class))

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
    def plot_and_save_synthetic_samples(synthetic_data):
        """
        Create a 4x15 plot of 60 synthetic samples and save it to a file.
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 15, figsize=(15, 8))
        axes = axes.flatten()

        for i in range(60):
            image = synthetic_data[i].permute(1, 2, 0).numpy()  # Convert CxHxW to HxWxC for Matplotlib
            axes[i].imshow(image)
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def perform_data_augmentation(dataset):
        data_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)
        ])
        return AugmentedSubset(dataset, transform=data_augmentation)

    def main(self):
        # Load training and test datasets
        train_dataset = self.load_dataset(train=True)
        test_dataset = self.load_dataset(train=False)
        all_el2n_scores, class_el2n_scores, _, class_accuracies, _, _, _, _ = self.load_results()

        # Compute hardness-based ratios and sample allocation
        ratios = self.compute_hardness_based_ratios(class_accuracies, class_el2n_scores)
        samples_per_class = self.compute_sample_allocation(ratios)

        # Perform resampling
        resampled_dataset = self.resample_dataset(train_dataset, all_el2n_scores, class_el2n_scores, samples_per_class)
        augmented_resampled_dataset = self.perform_data_augmentation(resampled_dataset)

        # Get DataLoaders
        resampled_loader = self.get_dataloader(augmented_resampled_dataset, shuffle=True)
        test_loader = self.get_dataloader(test_dataset, shuffle=False)

        # Print final sample allocation
        print("Samples per class after resampling in training set:")
        for class_id, count in samples_per_class.items():
            print(f"  Class {class_id}: {count}")

        model_save_dir = f"over_{self.oversampling_strategy}_under_{self.undersampling_strategy}_size_" \
                         f"{self.desired_dataset_size}_hardness_{self.hardness_estimation}"
        trainer = ModelTrainer(resampled_loader, test_loader, self.dataset_name, model_save_dir, False,
                               hardness='objective', clean_data=self.remove_noise)
        trainer.train_ensemble()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment3 with Data Resampling.")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="Name of the dataset (e.g., CIFAR10, CIFAR100, SVHN)")
    parser.add_argument('--desired_dataset_size', type=int, required=True,
                        help="Desired size of the dataset after resampling")
    parser.add_argument('--oversampling', type=str, required=True, choices=['random', 'easy', 'hard', 'SMOTE'],
                        help='Strategy used for oversampling (have to choose between `random`, `easy`, `hard`, and '
                             '`SMOTE`)')
    parser.add_argument('--undersampling', type=str, required=True, choices=['random', 'easy', 'hard', 'extreme'],
                        help='Strategy used for undersampling (have to choose between `random`, `prune_easy`, '
                             '`prune_hard`, and `prune_extreme`)')
    parser.add_argument('--class_hardness_estimation', type=str, required=True,
                        choices=['accuracy', 'EL2N', 'safe_pruning'],
                        help='Strategy used to estimate hardness of each class. The obtained ratio is used when '
                             'introducing imbalance to the training set. Choose between `accuracy`, `EL2N`, and'
                             '`safe_pruning`.')
    parser.add_argument('--remove_noise', action='store_true', help='Raise this flag to remove noise from the data.')
    args = parser.parse_args()

    experiment = Experiment3(args.dataset_name, args.desired_dataset_size, args.oversampling, args.undersampling,
                             args.class_hardness_estimation, args.remove_noise)
    experiment.main()
