"""This is the third important module that allows training the ensembles on resampled datasets."""

import argparse
import os
import pickle

import numpy as np
import tqdm

from config import get_config, ROOT
from data import get_dataloader, load_dataset, perform_data_augmentation
from data_resampling import DataResampling
from train_ensemble import ModelTrainer
from utils import compute_sample_allocation_after_resampling, load_hardness_estimates, set_reproducibility


class Experiment3:
    """Encapsulates all the necessary methods to perform the resampling experiments."""
    def __init__(self, dataset_name: str, oversampling: str, undersampling: str, hardness_estimator: str,
                 remove_noise: bool, alpha: int):
        """Initialize the Experiment3 class with configuration specific to the current experiment.

        :param dataset_name: Name of the dataset.
        :param oversampling: Name of the oversampling strategy. The viable options are 'random', 'easy', 'hard',
        'SMOTE', 'rEDM', 'hEDM', 'aEDM', and 'none', with the last one indicating no oversampling (used for
        ablation study).
        :param hardness_estimator: Name of the hardness estimator. Uses AUM by default.
        :param remove_noise: Flag indicating whether the experiments are conducted on dataset that had noise removal
        applied to it or not (see experiment1.py). This is only useful for random oversampling and SMOTE as these method
        are the most heavily impacted by label noise. That is not to say generative models aren't, it's just that we
        don't have resources to retrain a diffusion model on `clean` datasets.
        :param alpha: Integer used for computing resampling ratio thst allows to modify the degree of introduced data
        imbalance.
        """
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

    def main(self):
        """Main function that runs the code."""
        _, training_dataset, _, test_dataset = load_dataset(self.dataset_name, self.data_cleanliness == 'clean')
        labels = [training_dataset[idx][1].item() for idx in range(len(training_dataset))]

        hardness_estimates = load_hardness_estimates(self.data_cleanliness, self.dataset_name)
        hardness_over_models = [hardness_estimates[(0, model_id)][self.hardness_estimator]
                                for model_id in range(len(hardness_estimates))]
        hardness_estimates = np.mean(np.array(hardness_over_models[:self.num_models_for_hardness]), axis=0)

        samples_per_class, hardnesses_by_class = compute_sample_allocation_after_resampling(hardness_estimates, labels,
                                                                                            self.num_classes,
                                                                                            self.num_samples,
                                                                                            self.hardness_estimator,
                                                                                            alpha=self.alpha)
        with open(os.path.join(self.hardness_save_dir, f'alpha_{self.alpha}', 'samples_per_class.pkl'), 'wb') as file:
            # noinspection PyTypeChecker
            pickle.dump(samples_per_class, file)

        high_is_hard = self.hardness_estimator not in ['Confidence', 'AUM']
        actual_counts, resampled_loaders = None, []
        for _ in tqdm.tqdm(range(self.dataset_count)):
            resampler = DataResampling(training_dataset, self.num_classes, self.oversampling_strategy,
                                       self.undersampling_strategy, hardnesses_by_class, high_is_hard,
                                       self.dataset_name, self.num_models_for_hardness, self.mean, self.std)
            resampled_dataset = resampler.resample_data(samples_per_class)
            # Sanity check below
            labels = [resampled_dataset[idx][1].item() for idx in range(len(resampled_dataset))]
            actual_counts = np.bincount(np.array(labels))
            if self.undersampling_strategy != 'none' and self.oversampling_strategy != 'none':
                for cls in range(self.num_classes):
                    assert actual_counts[cls] == samples_per_class[cls], \
                        f"Mismatch for class {cls}: allocated {samples_per_class[cls]}, got {actual_counts[cls]}"

            augmented_resampled_dataset = perform_data_augmentation(resampled_dataset, self.dataset_name)
            resampled_loaders.append(get_dataloader(augmented_resampled_dataset, batch_size=self.config['batch_size'],
                                                    shuffle=True))
        test_loader = get_dataloader(test_dataset, batch_size=self.config['batch_size'])

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
                        choices=['random', 'easy', 'hard', 'SMOTE', 'rEDM', 'hEDM', 'aEDM', 'none'],
                        help='Strategy used for oversampling (have to choose between `random`, `easy`, `hard`, '
                             '`SMOTE``, `rEDM`, `hEDM`, `aEDM`, and `none`).')
    parser.add_argument('--undersampling', type=str, required=True, choices=['easy', 'none'],
                        help='Strategy used for undersampling (have to choose between `random`, `prune_easy`, '
                             '`prune_hard`, `prune_extreme`, and `none`).')
    parser.add_argument('--hardness_estimator', type=str, default='AUM',
                        help='Specifies which instance level hardness estimator to use.')
    parser.add_argument('--remove_noise', action='store_true', help='Raise this flag to remove noise from the data.')
    parser.add_argument('--alpha', type=int, default=1, help='Used to control the degree of introduced imbalance.')
    args = parser.parse_args()

    experiment = Experiment3(**vars(args))
    experiment.main()