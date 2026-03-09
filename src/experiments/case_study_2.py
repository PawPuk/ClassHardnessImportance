"""This is the third important module that allows training the ensembles on resampled datasets."""

import argparse
import os
import pickle

import tqdm

from src.config.config import get_config, ROOT
from src.data.loading import get_dataloader, load_dataset, perform_data_augmentation
from src.resampling.data_resampling import DataResampling
from src.training.train_ensemble import ModelTrainer
from src.utils.evaluation import compute_sample_allocation_after_resampling
from src.utils.io import load_hardness_estimates
from src.utils.reproducibility import set_reproducibility


class Experiment3:
    """Encapsulates all the necessary methods to perform the resampling experiments."""
    def __init__(self, dataset_name: str, oversampling: str, undersampling: str, hardness_estimator: str, alpha: int):
        """Initialize the Experiment3 class with configuration specific to the current experiment.

        :param dataset_name: Name of the dataset.
        :param oversampling: Name of the oversampling strategy. The viable options are 'random', 'SMOTE', 'rEDM',
        'hEDM', 'aEDM', and 'none', with the last one indicating no oversampling (used for ablation study).
        :param hardness_estimator: Name of the hardness estimator that will be used to guide pruning.
        :param alpha: Integer used for computing resampling ratio that allows to modify the degree of introduced data
        imbalance.
        """
        self.dataset_name = dataset_name
        self.oversampling_strategy = oversampling
        self.undersampling_strategy = undersampling
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

        self.hardness_save_dir = os.path.join(ROOT, "Results", self.dataset_name)
        self.figure_save_dir = os.path.join(ROOT, f"Figures/{self.dataset_name}_alpha{self.alpha}/")
        for save_dir in [self.figure_save_dir, os.path.join(self.hardness_save_dir, f'alpha_{self.alpha}')]:
            os.makedirs(save_dir, exist_ok=True)

    def main(self):
        """Main function that runs the code."""
        _, training_dataset, _, test_dataset = load_dataset(self.dataset_name)
        print(len(training_dataset))
        labels = [training_dataset[idx][1].item() for idx in range(len(training_dataset))]

        hardness_estimates = load_hardness_estimates(self.dataset_name, self.hardness_estimator,
                                                     self.num_models_for_hardness)

        samples_per_class, hardness_by_class = compute_sample_allocation_after_resampling(
            hardness_estimates, labels, self.num_classes, self.num_samples, self.hardness_estimator,
            alpha=self.alpha
        )
        # We get rid of the indices of hardness estimates as they are only required for pruning.
        hardness_by_class = {c: [hardness_by_class[c][i][1] for i in range(len(hardness_by_class[c]))]
                             for c in hardness_by_class.keys()}

        with open(os.path.join(self.hardness_save_dir, f'alpha_{self.alpha}', 'samples_per_class.pkl'), 'wb') as file:
            pickle.dump(samples_per_class, file)

        # For Confidence and AUM high scores do not indicate hard samples (unlike in other estimators)
        high_is_hard = self.hardness_estimator not in ['Confidence', 'AUM']
        actual_counts, resampled_loaders = None, []

        # Create `self.dataset_count` versions of resampled dataset to account for variability.
        for dataset_idx in tqdm.tqdm(range(self.dataset_count)):
            set_reproducibility(42 * dataset_idx)
            resampler = DataResampling(training_dataset, self.num_classes, self.oversampling_strategy,
                                       self.undersampling_strategy, hardness_by_class, high_is_hard,
                                       self.dataset_name, self.num_models_for_hardness, self.mean, self.std,
                                       self.num_epochs)
            resampled_dataset = resampler.resample_data(samples_per_class)
            print(f'Resampled dataset contains {len(resampled_dataset)} data samples.')

            augmented_resampled_dataset = perform_data_augmentation(resampled_dataset, self.dataset_name)
            resampled_loaders.append(get_dataloader(augmented_resampled_dataset, batch_size=self.config['batch_size'],
                                                    shuffle=True))
        test_loader = get_dataloader(test_dataset, batch_size=self.config['batch_size'])

        model_save_dir = (f"over_{self.oversampling_strategy}_under_{self.undersampling_strategy}_alpha_{self.alpha}_"
                          f"hardness_{self.hardness_estimator}")
        print(f'Training dataset contains {len(training_dataset)} data samples.')
        trainer = ModelTrainer(len(training_dataset), resampled_loaders, test_loader, self.dataset_name,
                               model_save_dir, False)
        trainer.train_ensemble()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment3 with Data Resampling.")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="Name of the dataset (e.g., CIFAR10, CIFAR100, SVHN).")
    parser.add_argument('--oversampling', type=str, required=True,
                        choices=['random', 'SMOTE', 'rEDM', 'hEDM', 'aEDM', 'none'],
                        help='Strategy used for oversampling (have to choose between `random`, `SMOTE``, `rEDM`, '
                             '`hEDM`, `aEDM`, and `none`). `none` allows ablation study')
    parser.add_argument('--undersampling', type=str, required=True, choices=['easy', 'none'],
                        help='Strategy used for undersampling (choose between `none` and `easy`). `none` was included '
                             'to allow ablation study).')
    parser.add_argument('--hardness_estimator', type=str, default='AUM',
                        help='Specifies which hardness estimator to use for pruning.')
    parser.add_argument('--alpha', type=float, default=1, help='Used to control the degree of introduced imbalance.')
    args = parser.parse_args()

    experiment = Experiment3(**vars(args))
    experiment.main()
