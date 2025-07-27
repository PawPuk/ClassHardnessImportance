import argparse
import os.path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pruning import DataPruning
from train_ensemble import ModelTrainer
from config import get_config, ROOT
from data import load_dataset
from utils import load_hardness_estimates, load_results, set_reproducibility


class Experiment2:
    def __init__(self, pruning_strategy: str, dataset_name: str, pruning_type: str, pruning_rate: int,
                 hardness_estimator: str):
        set_reproducibility()

        self.dataset_name = dataset_name
        self.pruning_strategy = pruning_strategy
        self.pruning_type = pruning_type
        self.pruning_rate = pruning_rate
        self.hardness_estimator = hardness_estimator

        # Constants taken from config
        config = get_config(dataset_name)
        self.BATCH_SIZE = config['batch_size']
        self.SAVE_EPOCH = config['save_epoch']
        self.MODEL_DIR = config['save_dir']
        self.NUM_CLASSES = config['num_classes']
        self.NUM_EPOCHS = config['num_epochs']

        self.results_save_dir = os.path.join(ROOT, 'Results/')

    def prune_dataset(self, labels, training_loader, hardness_estimates = None, high_is_hard = None,
                      imbalance_ratio = None):
        pruner = DataPruning(hardness_estimates, self.pruning_rate, self.dataset_name, high_is_hard, imbalance_ratio)

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

        if self.pruning_type == 'easy':
            hardness_estimates = load_hardness_estimates('unclean', self.dataset_name, self.hardness_estimator)
            if self.hardness_estimator in ['DataIQ', 'iDataIQ', 'Forgetting', 'Loss', 'iLoss', 'EL2N']:
                pruned_dataset = self.prune_dataset(labels, training_loader, hardness_estimates, True)
            elif self.hardness_estimator in ['Confidence', 'iConfidence', 'AUM', 'iAUM']:
                pruned_dataset = self.prune_dataset(labels, training_loader, hardness_estimates, False)
            else:
                raise ValueError(f'{self.hardness_estimator} is not a supported hardness estimator.')
        else:
            imbalance_ratio = load_results(os.path.join(self.results_save_dir, f'unclean{self.dataset_name}'))
            pruned_dataset = self.prune_dataset(labels, training_loader, imbalance_ratio = imbalance_ratio)

        # This is required to shuffle the data.
        pruned_training_loader = DataLoader(pruned_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=2)

        pruning_type = f"{self.pruning_type}_{self.pruning_strategy}{self.pruning_rate}"
        trainer = ModelTrainer(len(training_dataset), pruned_training_loader, test_loader, self.dataset_name,
                               pruning_type, False)
        trainer.train_ensemble()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EL2N Score Calculation and Dataset Pruning')
    parser.add_argument('--pruning_strategy', type=str, choices=['clp', 'dlp'],
                        help='Choose pruning strategy: clp (fixed class level pruning) or dlp (data level pruning)')
    parser.add_argument('--dataset_name', type=str, choices=['CIFAR10', 'CIFAR100'],
                        help='Specify the dataset name (default: CIFAR10)')
    parser.add_argument('--pruning_type', type=str, choices=['easy', 'random'],
                        help='Specify if the pruning is to be performed on easy or random samples.')
    parser.add_argument('--pruning_rate', type=int,
                        help='Percentage of data samples that will be removed during data pruning (use integers).')
    parser.add_argument('--hardness_estimator', type=str, default='AUM',
                        help='Specifies which hardness estimator to use for pruning.')

    args = parser.parse_args()

    # Initialize and run the experiment
    experiment = Experiment2(args.pruning_strategy, args.dataset_name,  args.pruning_type, args.pruning_rate,
                             args.hardness_estimator)
    experiment.run_experiment()
