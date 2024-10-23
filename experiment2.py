import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from data_pruning import DataPruning
from neural_networks import ResNet18LowRes
from utils import get_config
from train_ensemble import ModelTrainer


class Experiment2:
    def __init__(self, dataset_name, pruning_strategy, pruning_rate):
        self.dataset_name = dataset_name
        self.pruning_strategy = pruning_strategy
        self.pruning_rate = pruning_rate

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
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=self.BATCH_SIZE, shuffle=True,
                                                      num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=2)
        training_set_size = len(training_set)

        return training_loader, test_loader, training_set_size

    def create_model(self):
        model = ResNet18LowRes(num_classes=self.NUM_CLASSES)
        return model

    def compute_el2n(self, model, dataloader):
        model.eval()  # Set the model to evaluation mode
        el2n_scores = []  # Store EL2N scores for all training samples

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)

                # Apply softmax to model outputs
                softmax_outputs = F.softmax(outputs, dim=1)

                # One-hot encode the labels
                one_hot_labels = F.one_hot(labels, num_classes=self.NUM_CLASSES).float()

                # Compute the L2 norm of the error
                l2_errors = torch.norm(softmax_outputs - one_hot_labels, dim=1)  # L2 norm along the class dimension

                # Extend the list with L2 errors for this batch
                el2n_scores.extend(l2_errors.cpu().numpy())  # Convert to CPU and add to the list

        return el2n_scores

    def load_model_and_compute_el2n(self, model_id):
        model = self.create_model().cuda()
        # Construct the model path based on pruning strategy, dataset name, and save epoch
        model_path = os.path.join(self.MODEL_DIR, 'none', self.dataset_name,
                                  f'model_{model_id}_epoch_{self.SAVE_EPOCH}.pth')

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f'Model {model_id} loaded successfully from epoch {self.SAVE_EPOCH}.')

            # Compute EL2N scores for this model on the training set
            el2n_scores = self.compute_el2n(model, self.training_loader)
            return el2n_scores
        else:
            print(f'Model {model_id} not found at epoch {self.SAVE_EPOCH}.')
            return None

    def collect_el2n_scores(self):
        all_el2n_scores = [[] for _ in range(self.training_set_size)]  # List of lists to store scores from each model

        # Loop over all 10 models
        for model_id in range(10):
            el2n_scores = self.load_model_and_compute_el2n(model_id)
            if el2n_scores:
                # Store EL2N scores from this model into the master list
                for i in range(self.training_set_size):
                    all_el2n_scores[i].append(el2n_scores[i])

        return all_el2n_scores

    def group_scores_by_class(self, el2n_scores):
        class_el2n_scores = {i: [] for i in range(self.NUM_CLASSES)}  # Dictionary to store scores by class
        labels = []  # Store corresponding labels

        # Since we are not shuffling the data loader, we can directly match scores with their labels
        for i, (_, label) in enumerate(self.training_loader.dataset):
            class_el2n_scores[label].append(el2n_scores[i])
            labels.append(label)  # Collect the labels

        return class_el2n_scores, labels

    @staticmethod
    def compute_class_statistics(class_el2n_scores):
        class_stats = {}

        for class_id, scores in class_el2n_scores.items():
            scores_array = np.array(scores)  # Convert to numpy array for statistical analysis

            # Compute mean, std, and quartiles
            means = np.mean(scores_array, axis=1)
            q1 = np.percentile(means, 25)
            q3 = np.percentile(means, 75)
            min_val = np.min(means)
            max_val = np.max(means)

            class_stats[class_id] = {
                "q1": q1,
                "q3": q3,
                "min": min_val,
                "max": max_val
            }

        return class_stats

    def plot_class_level_candlestick(self, class_stats):
        # Prepare the saving directory and file name
        save_dir = os.path.join('Figures/', self.pruning_strategy, self.dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f'hardness_distribution.pdf')

        # Prepare the data for plotting
        class_ids = list(class_stats.keys())
        q1_values = [class_stats[class_id]["q1"] for class_id in class_ids]
        q3_values = [class_stats[class_id]["q3"] for class_id in class_ids]
        min_values = [class_stats[class_id]["min"] for class_id in class_ids]
        max_values = [class_stats[class_id]["max"] for class_id in class_ids]

        # Create the candlestick chart
        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(self.NUM_CLASSES):
            # Draw the candlestick (real body: Q1 to Q3, shadow: min to max)
            ax.plot([i, i], [min_values[i], max_values[i]], color='black')  # Shadow
            ax.plot([i, i], [q1_values[i], q3_values[i]], color='blue', lw=6)  # Real body

        ax.set_xticks(range(self.NUM_CLASSES))
        ax.set_xticklabels([f'Class {i}' for i in range(self.NUM_CLASSES)])
        ax.set_xlabel("Classes")
        ax.set_ylabel("EL2N Score (L2 Norm)")
        ax.set_title("Class-Level EL2N Scores Candlestick Plot")
        plt.savefig(file_name)

    def prune_dataset(self, el2n_scores, class_el2n_scores, labels):
        # Instantiate the DataPruning class with el2n_scores, class_el2n_scores, and labels
        pruner = DataPruning(el2n_scores, class_el2n_scores, labels, self.pruning_rate, self.dataset_name)

        if self.pruning_strategy == 'dlp':
            pruned_indices = pruner.dataset_level_pruning()
        elif self.pruning_strategy == 'fclp':
            pruned_indices = pruner.fixed_class_level_pruning()
        else:
            raise ValueError('Wrong value of the parameter `pruning_strategy`.')

        # Create a new pruned dataset
        pruned_dataset = torch.utils.data.Subset(self.training_loader.dataset, pruned_indices)

        return pruned_dataset

    def run_experiment(self):
        # Collect EL2N scores across all models for the training set
        all_el2n_scores = self.collect_el2n_scores()

        # Group EL2N scores by class and get labels
        class_el2n_scores, labels = self.group_scores_by_class(all_el2n_scores)

        # Compute class-level statistics for candlestick chart
        class_stats = self.compute_class_statistics(class_el2n_scores)

        # Plot the class-level candlestick chart
        self.plot_class_level_candlestick(class_stats)

        # Perform dataset-level pruning
        pruned_dataset = self.prune_dataset(all_el2n_scores, class_el2n_scores, labels)

        # Create data loader for pruned dataset
        pruned_training_loader = DataLoader(pruned_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=2)

        # Train ensemble of 10 models on pruned data (without saving probe models)
        trainer = ModelTrainer(pruned_training_loader, self.test_loader, self.dataset_name,
                               f'{self.pruning_strategy + str(self.pruning_rate)}', False)
        trainer.train_ensemble()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EL2N Score Calculation and Dataset Pruning')
    parser.add_argument('--pruning_strategy', type=str, default='dlp', choices=['fclp', 'dlp'],
                        help='Choose pruning strategy: fclp (fixed class level pruning) or dlp (data level pruning)')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='Specify the dataset name (default: CIFAR10)')
    parser.add_argument('--pruning_rate', type=int, default=50,
                        help='Percentage of data samples that will be removed during data pruning (use integers).')

    args = parser.parse_args()

    # Initialize and run the experiment
    experiment = Experiment2(args.dataset_name, args.pruning_strategy, args.pruning_rate)
    experiment.run_experiment()
