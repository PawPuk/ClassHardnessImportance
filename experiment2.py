import argparse
import os
from typing import List, Union

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader

from data_pruning import DataPruning
from train_ensemble import ModelTrainer
from config import get_config, ROOT
from data import load_dataset
from utils import load_aum_results, load_forgetting_results, load_results, set_reproducibility


class Experiment2:
    def __init__(self, pruning_strategy: str, dataset_name: str, pruning_rate: int, hardness_estimator: str):
        set_reproducibility()

        self.dataset_name = dataset_name
        self.pruning_strategy = pruning_strategy
        self.pruning_rate = pruning_rate
        self.hardness_estimator = hardness_estimator

        # Constants taken from config
        config = get_config(dataset_name)
        self.BATCH_SIZE = config['batch_size']
        self.SAVE_EPOCH = config['save_epoch']
        self.MODEL_DIR = config['save_dir']
        self.NUM_CLASSES = config['num_classes']
        self.NUM_EPOCHS = config['num_epochs']

    def prune_dataset(self, hardness_estimates: List[List[Union[float, int]]], labels: NDArray[int],
                      training_loader: DataLoader, high_is_hard: bool) -> torch.utils.data.Subset:
        pruner = DataPruning(hardness_estimates, self.pruning_rate, self.dataset_name, high_is_hard)

        if self.pruning_strategy == 'dlp':
            remaining_indices = pruner.dataset_level_pruning(labels)
        elif self.pruning_strategy == 'clp':
            remaining_indices = pruner.class_level_pruning(labels)
        else:
            raise ValueError('Wrong value of the parameter `pruning_strategy`.')

        # Create a new pruned dataset
        pruned_dataset = torch.utils.data.Subset(training_loader.dataset, remaining_indices)

        return pruned_dataset

    def run_experiment(self):
        training_loader, training_dataset, test_loader, _ = load_dataset(self.dataset_name, False, False, True)
        labels = np.array([label for _, label, _ in training_dataset])

        if self.hardness_estimator == 'AUM':
            aum_scores = load_aum_results('unclean', self.dataset_name, self.NUM_EPOCHS)
            pruned_dataset = self.prune_dataset(aum_scores, labels, training_loader, False)
        elif self.hardness_estimator == 'Forgetting':
            forgetting_scores = load_forgetting_results('unclean', self.dataset_name)
            pruned_dataset = self.prune_dataset(forgetting_scores, labels, training_loader, True)
        elif self.hardness_estimator == 'EL2N':
            el2n_path = os.path.join(ROOT, f'Results/unclean{self.dataset_name}/el2n_scores.pkl')
            el2n_scores = load_results(el2n_path)
            pruned_dataset = self.prune_dataset(el2n_scores, labels, training_loader, True)
        else:
            raise ValueError(f'{self.hardness_estimator} is not a supported hardness estimator.')

        # This is required to shuffle the data.
        pruned_training_loader = DataLoader(pruned_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=2)

        pruning_type = f"{self.pruning_strategy}{self.pruning_rate}"
        trainer = ModelTrainer(len(training_dataset), pruned_training_loader, test_loader, self.dataset_name,
                               pruning_type, False)
        trainer.train_ensemble()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EL2N Score Calculation and Dataset Pruning')
    parser.add_argument('--pruning_strategy', type=str, choices=['clp', 'dlp'],
                        help='Choose pruning strategy: clp (fixed class level pruning) or dlp (data level pruning)')
    parser.add_argument('--dataset_name', type=str, choices=['CIFAR10', 'CIFAR100'],
                        help='Specify the dataset name (default: CIFAR10)')
    parser.add_argument('--pruning_rate', type=int,
                        help='Percentage of data samples that will be removed during data pruning (use integers).')
    parser.add_argument('--hardness_estimator', type=str, choices=['AUM', 'EL2N', 'Forgetting'],
                        help='Specifies which hardness estimator to use for pruning.')

    args = parser.parse_args()

    # Initialize and run the experiment
    experiment = Experiment2(args.pruning_strategy, args.dataset_name,  args.pruning_rate, args.hardness_estimator)
    experiment.run_experiment()

