"""This is the second important module that allows training the ensembles on pruned subdatasets."""

import argparse
from copy import deepcopy
import os.path
from typing import Dict, List, Tuple, Union

from torch.utils.data import DataLoader

from src.config.config import get_config, ROOT
from src.data.datasets import AugmentedSubset, IndexedDataset
from src.data.loading import load_dataset, perform_data_augmentation
from src.pruning.data_pruning import DataPruning
from src.training.train_ensemble import ModelTrainer
from src.utils.evaluation import compute_sample_allocation_after_resampling
from src.utils.io import load_hardness_estimates
from src.utils.reproducibility import set_reproducibility


class Experiment2:
    """Encapsulates all the necessary methods to perform the pruning experiments."""
    def __init__(
            self,
            dataset_name: str,
            pruning_rate: int,
            hardness_estimator: str,
            oversampling_strategy: str
    ):
        """Initialize the Experiment2 class with configuration specific to the current experiment.

        :param dataset_name: Name of the dataset.
        :param pruning_rate: An integer specifying the percentage of samples that will be pruned from the dataset.
        :param hardness_estimator: Name of the hardness estimator that will be used to compute the resampling
        ratios and guide pruning.
        :param oversampling_strategy: Name of the oversampling strategy. If set to none than no resampling will be
        applied after pruning.
        """
        self.dataset_name = dataset_name
        self.pruning_rate = pruning_rate
        self.hardness_estimator = hardness_estimator
        self.oversampling_strategy = oversampling_strategy

        # Constants taken from config
        config = get_config(dataset_name)
        self.BATCH_SIZE = config['batch_size']
        self.SAVE_EPOCH = config['save_epoch']
        self.MODEL_DIR = config['save_dir']
        self.NUM_CLASSES = config['num_classes']
        self.NUM_EPOCHS = config['num_epochs']
        self.NUM_TRAINING_SAMPLES = sum(config['num_training_samples'])
        self.DATASET_COUNT = config['num_datasets']
        self.NUM_MODELS_FOR_HARDNESS = config['num_models_for_hardness']

        self.figure_save_dir = os.path.join(ROOT, 'Figures/')

    def prune_dataset(
            self, labels: List[int],
            training_dataset: Union[AugmentedSubset, IndexedDataset],
            samples_per_class: List[int],
            hardness_sorted_by_class: Dict[int, List[Tuple[int, float]]]
    ) -> AugmentedSubset:
        """Produce the pruned dataset through DataPruning"""
        pruner = DataPruning(self.pruning_rate, self.dataset_name, samples_per_class, self.hardness_estimator)
        pruned_subdataset = pruner.prune_and_resample(self.oversampling_strategy, labels, training_dataset,
                                                      deepcopy(hardness_sorted_by_class))
        augmented_subdataset = perform_data_augmentation(pruned_subdataset, self.dataset_name)
        return augmented_subdataset

    def run_experiment(self):
        """Main method for running the experiments."""
        _, training_dataset, test_loader, _ = load_dataset(self.dataset_name)
        labels = [training_dataset[idx][1].item() for idx in range(len(training_dataset))]

        hardness_estimates = load_hardness_estimates(self.dataset_name, self.hardness_estimator,
                                                     self.NUM_MODELS_FOR_HARDNESS)
        samples_per_class, hardness_by_class = compute_sample_allocation_after_resampling(
            hardness_estimates, labels, self.NUM_CLASSES, self.NUM_TRAINING_SAMPLES, self.hardness_estimator,
            self.pruning_rate
        )

        pruned_training_loaders = []
        for dataset_idx in range(self.DATASET_COUNT):
            set_reproducibility(42 * dataset_idx)
            pruned_dataset = self.prune_dataset(labels, training_dataset, samples_per_class, hardness_by_class)
            pruned_training_loaders.append(DataLoader(pruned_dataset, batch_size=self.BATCH_SIZE, shuffle=True,
                                                      num_workers=2))

        model_dir = f"{self.oversampling_strategy}_pruning_rate_{self.pruning_rate}"
        trainer = ModelTrainer(len(training_dataset), pruned_training_loaders, test_loader, self.dataset_name,
                               model_dir, False)
        trainer.train_ensemble()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EL2N Score Calculation and Dataset Pruning')
    parser.add_argument('--dataset_name', type=str, choices=['CIFAR10', 'CIFAR100'],
                        help='Specify the dataset name (default: CIFAR10)')
    parser.add_argument('--pruning_rate', type=int,
                        help='Percentage of data samples that will be removed during data pruning (use integers).')
    parser.add_argument('--hardness_estimator', type=str, default='AUM',
                        help='Specifies which hardness estimator to use for computing resampling ratios.')
    parser.add_argument('--oversampling_strategy', type=str, choices=['none', 'random', 'SMOTE', 'holdout'],
                        help='Specifies what oversampling to use. If set to `none` then no resampling will be used '
                             'after pruning (useful for ablation study and benchmarking)')

    args = parser.parse_args()

    # Initialize and run the experiment
    experiment = Experiment2(args.dataset_name, args.pruning_rate, args.hardness_estimator, args.oversampling_strategy)
    experiment.run_experiment()
