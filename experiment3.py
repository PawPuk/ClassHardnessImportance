import argparse
import os
import pickle
import random

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from config import get_config, ROOT
from data import AugmentedSubset, load_dataset
from data_pruning import DataResampling
from train_ensemble import ModelTrainer
from utils import set_reproducibility, load_hardness_estimates


class Experiment3:
    def __init__(self, dataset_name, oversampling, undersampling, hardness_estimator, remove_noise, alpha):
        self.dataset_name = dataset_name
        self.oversampling_strategy = oversampling
        self.undersampling_strategy = undersampling
        self.data_cleanliness = 'clean' if remove_noise else 'unclean'
        self.hardness_estimator = hardness_estimator
        self.alpha = alpha

        self.config = get_config(dataset_name)
        self.num_classes = self.config['num_classes']
        self.num_epochs = self.config['num_epochs']
        self.num_samples = sum(self.config['num_training_samples'])
        self.num_models_for_hardness = self.config['num_models_for_hardness']
        self.mean = self.config['mean']
        self.std = self.config['std']
        self.dataset_count = self.config['num_datasets']

        self.hardness_save_dir = os.path.join(ROOT, f"Results/{self.data_cleanliness}{self.dataset_name}/")
        self.figure_save_dir = os.path.join(ROOT, f"Figures/{self.dataset_name}_alpha{self.alpha}/")
        for save_dir in [self.figure_save_dir, os.path.join(self.hardness_save_dir, f'alpha_{self.alpha}')]:
            os.makedirs(save_dir, exist_ok=True)

    def load_hardness_estimates(self):
        hardness_estimates = list(load_hardness_estimates(self.data_cleanliness, self.dataset_name).values())
        hardness_over_models = [hardness_estimates[model_id][self.hardness_estimator]
                                for model_id in range(len(hardness_estimates))]

        hardness_of_ensemble = np.mean(hardness_over_models[:self.num_models_for_hardness], axis=0)
        return hardness_of_ensemble

    def compute_sample_allocation(self, hardness_scores, dataset):
        """
        Compute hardness-based ratios based on class-level accuracies.
        """
        hardnesses_by_class, hardness_of_classes = {class_id: [] for class_id in range(self.num_classes)}, {}

        for i, (_, label, _) in enumerate(dataset):
            hardnesses_by_class[label.item()].append(hardness_scores[i])

        for label in range(self.num_classes):
            if self.hardness_estimator in ['AUM', 'Confidence', 'iAUM', 'iConfidence']:
                hardness_of_classes[label] = 1 / np.mean(hardnesses_by_class[label])
            else:
                hardness_of_classes[label] = np.mean(hardnesses_by_class[label])

        ratios = {class_id: class_hardness / sum(hardness_of_classes.values())
                  for class_id, class_hardness in hardness_of_classes.items()}
        samples_per_class = {class_id: int(round(ratio * self.num_samples)) for class_id, ratio in ratios.items()}

        average_sample_count = int(np.mean(list(samples_per_class.values())))
        for class_id in samples_per_class.keys():
            absolute_difference = abs(samples_per_class[class_id] - average_sample_count)
            if samples_per_class[class_id] > average_sample_count:
                samples_per_class[class_id] = average_sample_count + int(self.alpha * absolute_difference)
            else:
                samples_per_class[class_id] = average_sample_count - int(self.alpha * absolute_difference)

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

    def perform_data_augmentation(self, resampled_dataset: AugmentedSubset) -> AugmentedSubset:
        # The resampled_dataset has already been normalized by load_dataset so no need for ToTensor() and Normalize().
        if self.dataset_name in ['CIFAR100', 'CIFAR10']:
            data_augmentation = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        else:
            raise ValueError('Unsupported dataset.')
        return AugmentedSubset(resampled_dataset, transform=data_augmentation)

    def main(self):
        # The value of the shuffle parameter below does not matter as we don't use the loaders.
        _, training_dataset, _, test_dataset = load_dataset(self.dataset_name, self.data_cleanliness == 'clean', False,
                                                            False)
        hardness_scores = self.load_hardness_estimates()

        hardnesses_by_class, samples_per_class = self.compute_sample_allocation(hardness_scores, training_dataset)
        with open(os.path.join(self.hardness_save_dir, f'alpha_{self.alpha}', 'samples_per_class.pkl'), 'wb') as file:
            pickle.dump(samples_per_class, file)

        high_is_hard = self.hardness_estimator in ['Confidence', 'AUM']
        actual_counts, resampled_loaders = None, []
        for _ in range(self.dataset_count):
            resampler = DataResampling(training_dataset, self.num_classes, self.oversampling_strategy,
                                       self.undersampling_strategy, hardnesses_by_class, high_is_hard,
                                       self.dataset_name, self.num_models_for_hardness, self.mean, self.std)
            resampled_dataset = resampler.resample_data(samples_per_class)
            # Sanity check below
            labels = [lbl for _, lbl, _ in resampled_dataset]
            actual_counts = np.bincount(np.array(labels))
            if self.undersampling_strategy != 'none' and self.oversampling_strategy != 'none':
                for cls in range(self.num_classes):
                    assert actual_counts[cls] == samples_per_class[cls], \
                        f"Mismatch for class {cls}: allocated {samples_per_class[cls]}, got {actual_counts[cls]}"

            augmented_resampled_dataset = self.perform_data_augmentation(resampled_dataset)
            resampled_loaders.append(self.get_dataloader(augmented_resampled_dataset, shuffle=True))
        test_loader = self.get_dataloader(test_dataset, shuffle=False)

        print("Samples per class after resampling in training set:")
        for class_id, count in enumerate(actual_counts):
            print(f"  Class {class_id}: {count}")

        model_save_dir = (f"over_{self.oversampling_strategy}_under_{self.undersampling_strategy}_alpha_{self.alpha}_"
                          f"hardness_{self.hardness_estimator}")
        trainer = ModelTrainer(len(training_dataset), resampled_loaders, test_loader, self.dataset_name,
                               model_save_dir, False, clean_data=self.data_cleanliness == 'clean')
        trainer.train_ensemble()


if __name__ == "__main__":
    set_reproducibility()

    parser = argparse.ArgumentParser(description="Experiment3 with Data Resampling.")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="Name of the dataset (e.g., CIFAR10, CIFAR100, SVHN).")
    parser.add_argument('--oversampling', type=str, required=True,
                        choices=['random', 'easy', 'hard', 'SMOTE', 'DDPM', 'rEDM', 'hEDM', 'aEDM', 'none'],
                        help='Strategy used for oversampling (have to choose between `random`, `easy`, `hard`, '
                             '`SMOTE`, `DDPM`, `rEDM`, `hEDM`, `aEDM`, and `none`).')
    parser.add_argument('--undersampling', type=str, required=True, choices=['easy', 'none'],
                        help='Strategy used for undersampling (have to choose between `random`, `prune_easy`, '
                             '`prune_hard`, `prune_extreme`, and `none`).')
    parser.add_argument('--hardness_estimator', type=str, default='AUM',
                        help='Specifies which instance level hardness estimator to use.')
    parser.add_argument('--remove_noise', action='store_true', help='Raise this flag to remove noise from the data.')
    parser.add_argument('--alpha', type=int, default=1, help='Used to control the degree of introduced imbalance.')
    args = parser.parse_args()
    if args.oversampling == 'DDPM' and args.dataset_name != 'CIFAR10':
        raise Exception('DDPM can only be used with CIFAR10.')

    experiment = Experiment3(**vars(args))
    experiment.main()
