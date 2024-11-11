import argparse
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from data_pruning import DataPruning
from utils import get_config
from train_ensemble import ModelTrainer


class Experiment2:
    def __init__(self, dataset_name, pruning_strategy, pruning_rate, scaling_type, protect_prototypes, hardness_type):
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        self.dataset_name = dataset_name
        self.pruning_strategy = pruning_strategy
        self.pruning_rate = pruning_rate
        self.scaling_type = scaling_type
        self.protect_prototypes = protect_prototypes
        self.hardness_type = hardness_type
        self.results_save_dir = os.path.join('Results/', self.dataset_name)

        # Constants taken from config
        config = get_config(dataset_name)
        self.BATCH_SIZE = config['batch_size']
        self.SAVE_EPOCH = config['save_epoch']
        self.MODEL_DIR = config['save_dir']
        self.NUM_CLASSES = config['num_classes']

        # Load dataset depending on dataset name
        self.training_loader, self.test_loader, self.training_set_size = self.load_dataset(self.dataset_name)

    def load_dataset(self, dataset_name):
        """
        Load the dataset based on the dataset_name and return the corresponding train and test loaders.

        :param dataset_name: The name of the dataset ('CIFAR10', 'CIFAR100', etc.)
        :return: (training_loader, test_loader, training_set_size)
        """
        if dataset_name == 'CIFAR10':
            # Transformations for CIFAR-10 dataset (Training and Test sets)
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            # CIFAR-10 Dataset (Training and Test sets)
            training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                        transform=train_transform)
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        elif dataset_name == 'CIFAR100':
            # Transformations for CIFAR-100 dataset (Training and Test sets)
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            # CIFAR-100 Dataset (Training and Test sets)
            training_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                         transform=train_transform)
            test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                     transform=test_transform)

        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        # Load training and test data
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=self.BATCH_SIZE, shuffle=False,
                                                      num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=2)
        training_set_size = len(training_set)

        return training_loader, test_loader, training_set_size

    def load_el2n_scores(self):
        with open(os.path.join(self.results_save_dir, 'el2n_scores.pkl'), 'rb') as file:
            return pickle.load(file)

    def prune_dataset(self, el2n_scores, class_el2n_scores, labels):
        # Instantiate the DataPruning class with el2n_scores, class_el2n_scores, and labels
        pruner = DataPruning(el2n_scores, class_el2n_scores, labels, self.pruning_rate, self.dataset_name,
                             self.protect_prototypes)

        if self.pruning_strategy == 'dlp':
            pruned_indices = pruner.dataset_level_pruning()
        elif self.pruning_strategy == 'fclp':
            pruned_indices = pruner.fixed_class_level_pruning()
        elif self.pruning_strategy == 'aclp':
            pruned_indices = pruner.adaptive_class_level_pruning(self.scaling_type)
        elif self.pruning_strategy == 'loop':
            pruned_indices = pruner.leave_one_out_pruning()
        else:
            raise ValueError('Wrong value of the parameter `pruning_strategy`.')

        # Create a new pruned dataset
        pruned_dataset = torch.utils.data.Subset(self.training_loader.dataset, pruned_indices)

        return pruned_dataset

    def run_experiment(self):
        # Collect EL2N scores across all models for the training set
        all_el2n_scores, class_el2n_scores, labels = self.load_el2n_scores()

        # Perform dataset-level pruning
        pruned_dataset = self.prune_dataset(all_el2n_scores, class_el2n_scores, labels)

        # Create data loader for pruned dataset
        pruned_training_loader = DataLoader(pruned_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=2)

        # Train ensemble on pruned data (without saving probe models)
        model_save_dir = f"{['unprotected', 'protected'][self.protect_prototypes]}_{self.hardness_type}_"
        if self.pruning_strategy == 'aclp':
            model_save_dir += f'{self.scaling_type}_'
        model_save_dir += f"{self.pruning_strategy}{self.pruning_rate}"
        trainer = ModelTrainer(pruned_training_loader, self.test_loader, self.dataset_name, model_save_dir, False)
        trainer.train_ensemble()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EL2N Score Calculation and Dataset Pruning')
    parser.add_argument('--pruning_strategy', type=str, default='dlp', choices=['fclp', 'dlp', 'aclp', 'loop'],
                        help='Choose pruning strategy: fclp (fixed class level pruning) or dlp (data level pruning)')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='Specify the dataset name (default: CIFAR10)')
    parser.add_argument('--pruning_rate', type=int, default=50,
                        help='Percentage of data samples that will be removed during data pruning (use integers).')
    parser.add_argument('--scaling_type', type=str, default='linear',
                        choices=['linear', 'exponential', 'inverted_exponential'],
                        help='Choose scaling type for adaptive class-level pruning: linear, exponential, '
                             'inverted_exponential')
    parser.add_argument('--protect_prototypes', action='store_true',
                        help="Raise this flag to protect the prototypes from pruning - don't prune 1% of the easiest "
                             "samples.")
    parser.add_argument('--hardness_type', type=str, choices=['objective', 'subjective'],
                        help="If set to 'subjective', each model will use the hardness of probe network obtained using "
                             "the same seed (similar to self-paced learning). For 'objective', the average hardness "
                             "computed using all probe networks is used (similar to transfer learning).")

    args = parser.parse_args()

    # Initialize and run the experiment
    experiment = Experiment2(args.dataset_name, args.pruning_strategy, args.pruning_rate, args.scaling_type,
                             args.protect_prototypes, args.hardness_type)
    experiment.run_experiment()

