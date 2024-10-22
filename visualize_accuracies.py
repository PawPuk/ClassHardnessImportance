from collections import defaultdict
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neural_networks import ResNet18LowRes


class ModelEvaluator:
    def __init__(self, base_dir='./Models/'):
        """
        Initialize the ModelEvaluator class.

        :param base_dir: Base directory where the saved models are located.
        """
        self.base_dir = base_dir
        self.baseline_results = {}  # To store baseline accuracies for comparison

    def find_saved_model_paths(self):
        """
        Find all model paths that were saved after 200 epochs in the directory structure, categorized by pruning
        strategy and dataset used.

        :return: Dictionary where keys are tuples of (pruning_type, dataset_name) and values are lists of paths to saved
        models.
        """
        saved_model_paths = defaultdict(list)

        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.pth') and 'epoch_200' in file:
                    path_parts = root.split(os.sep)
                    if len(path_parts) >= 3:
                        pruning_type = path_parts[-3]  # 'Models/pruning_type/dataset_name/...'
                        dataset_name = path_parts[-2]
                        saved_model_paths[(pruning_type, dataset_name)].append(os.path.join(root, file))

        return saved_model_paths

    @staticmethod
    def load_model(model_path):
        """
        Load a ResNet-18 model from the given path.

        :param model_path: Path to the saved model.
        :return: Loaded model.
        """
        model = ResNet18LowRes(num_classes=10).cuda()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    @staticmethod
    def load_dataset(dataset_name):
        """
        Load the dataset based on the dataset name.

        :param dataset_name: Name of the dataset (e.g., CIFAR10, CIFAR100).
        :return: Test data loader.
        """
        if dataset_name == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
            ])
            test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        elif dataset_name == 'CIFAR100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])
            test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        return test_loader

    @staticmethod
    def evaluate_model(model, test_loader):
        """
        Evaluate the model on the test set and return the accuracy and class-level accuracies.

        :param model: The trained model to evaluate.
        :param test_loader: DataLoader for the test set.
        :return: (overall accuracy, class accuracies).
        """
        class_correct, class_total = np.zeros(10), np.zeros(10)
        total_correct, total_samples = 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                for i in range(10):  # Assuming 10 classes
                    class_mask = (labels == i)
                    class_correct[i] += (predicted[class_mask] == i).sum().item()
                    class_total[i] += class_mask.sum().item()

        overall_accuracy = 100 * total_correct / total_samples
        class_accuracies = 100 * class_correct / class_total  # Per-class accuracies
        return overall_accuracy, class_accuracies

    def evaluate_ensemble(self, model_paths, test_loader):
        """
        Evaluate the ensemble of models and compute the average and std for the dataset-level accuracy and class-level
        accuracies.

        :param model_paths: List of paths to saved models.
        :param test_loader: DataLoader for the test set.
        :return: (avg_accuracy, std_accuracy, avg_class_accuracies, std_class_accuracies).
        """
        accuracies, class_accuracies = [], []

        for model_path in model_paths:
            model = self.load_model(model_path)
            accuracy, class_acc = self.evaluate_model(model, test_loader)
            accuracies.append(accuracy)
            class_accuracies.append(class_acc)

        # Convert to numpy for easy computation
        accuracies, class_accuracies = np.array(accuracies), np.array(class_accuracies)
        # Compute mean and std
        avg_accuracy, std_accuracy = np.mean(accuracies), np.std(accuracies)
        avg_class_accuracies = np.mean(class_accuracies, axis=0)
        std_class_accuracies = np.std(class_accuracies, axis=0)

        return avg_accuracy, std_accuracy, avg_class_accuracies, std_class_accuracies

    @staticmethod
    def plot_ensemble_results(avg_class_accuracies, std_class_accuracies, avg_accuracy, std_accuracy,
                              dataset_name, pruning_type):
        """
        Plot the ensemble results as candlestick plot for class-level accuracies and fill_between for dataset-level accuracy.

        :param avg_class_accuracies: Array of average class-level accuracies.
        :param std_class_accuracies: Array of std class-level accuracies.
        :param avg_accuracy: Average dataset-level accuracy.
        :param std_accuracy: Std of dataset-level accuracy.
        :param dataset_name: The name of the dataset.
        :param pruning_type: The pruning type used.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot candlesticks for class-level accuracies
        for i in range(10):  # Assuming 10 classes
            ax.errorbar(i + 1, avg_class_accuracies[i], yerr=std_class_accuracies[i], fmt='o', capsize=5,
                        label=f'Class {i + 1}')

        # Plot dataset-level accuracy as a filled region
        x = np.arange(1, 11)
        ax.fill_between(x, avg_accuracy - std_accuracy, avg_accuracy + std_accuracy, color='gray', alpha=0.3,
                        label='Dataset Accuracy')
        ax.plot(x, [avg_accuracy] * 10, color='black', linestyle='--')

        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Ensemble Results for {dataset_name} with {pruning_type}')
        ax.set_xticks(x)
        ax.legend()
        save_dir = os.path.join('Figures/', pruning_type, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f'ensemble_accuracies.pdf')
        plt.savefig(file_name)

    @staticmethod
    def plot_accuracy_diff(class_accuracy_diff, dataset_accuracy_diff, dataset_name, pruning_type):
        """
        Plot the difference in accuracy between pruned and baseline models as a bar chart for class-level accuracy
        and a horizontal line for dataset-level accuracy.

        :param class_accuracy_diff: Array of differences in class-level accuracies.
        :param dataset_accuracy_diff: Difference in dataset-level accuracy.
        :param dataset_name: The name of the dataset.
        :param pruning_type: The pruning type used.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot class-level accuracy differences as bars
        x = np.arange(10)  # Assuming 10 classes
        ax.bar(x, class_accuracy_diff, color='b', alpha=0.7)

        # Plot dataset-level accuracy difference as a horizontal line
        ax.axhline(y=dataset_accuracy_diff, color='r', linestyle='--', label=f'Dataset accuracy difference: {dataset_accuracy_diff:.2f}%')

        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy Difference (%)')
        ax.set_title(f'Accuracy Difference (Pruned vs Baseline) for {dataset_name} with {pruning_type}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {i+1}' for i in x])
        ax.legend()
        save_dir = os.path.join('Figures/', pruning_type, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f'ensemble_accuracy_improvements.pdf')
        plt.savefig(file_name)

    def evaluate_saved_models(self):
        """
        Iterate through all saved models grouped by pruning strategy and dataset, load them, and evaluate the ensemble.
        """
        saved_model_groups = self.find_saved_model_paths()
        # First, evaluate baseline (pruning_type == 'none') models
        for (pruning_type, dataset_name), model_paths in saved_model_groups.items():
            if pruning_type == 'none':
                print(f'\nEvaluating baseline for Dataset: {dataset_name}')
                test_loader = self.load_dataset(dataset_name)
                # Evaluate the ensemble
                avg_accuracy, std_accuracy, avg_class_accuracies, std_class_accuracies = self.evaluate_ensemble(
                    model_paths, test_loader)

                self.baseline_results[dataset_name] = {
                    'avg_accuracy': avg_accuracy,
                    'avg_class_accuracies': avg_class_accuracies
                }
                # Print overall ensemble accuracy
                print(f'Average dataset accuracy (baseline): {avg_accuracy:.2f}% (±{std_accuracy:.2f}%)')
                # Plot the results
                self.plot_ensemble_results(avg_class_accuracies, std_class_accuracies, avg_accuracy, std_accuracy,
                                           dataset_name, pruning_type)

        # Now evaluate the pruned models and compare them to baseline
        for (pruning_type, dataset_name), model_paths in saved_model_groups.items():
            if pruning_type != 'none':  # Evaluate pruned models only
                print(f'\nEvaluating ensemble for Dataset: {dataset_name}, Pruning Type: {pruning_type}')

                test_loader = self.load_dataset(dataset_name)
                avg_accuracy, std_accuracy, avg_class_accuracies, std_class_accuracies = self.evaluate_ensemble(
                    model_paths, test_loader)

                # Plot the original ensemble results for the pruned model
                self.plot_ensemble_results(avg_class_accuracies, std_class_accuracies, avg_accuracy, std_accuracy,
                                           dataset_name, pruning_type)

                # Get baseline results for comparison
                if dataset_name in self.baseline_results:
                    baseline_accuracy = self.baseline_results[dataset_name]['avg_accuracy']
                    baseline_class_accuracies = self.baseline_results[dataset_name]['avg_class_accuracies']

                    # Compute the difference between pruned and baseline
                    accuracy_diff = avg_accuracy - baseline_accuracy
                    class_accuracy_diff = avg_class_accuracies - baseline_class_accuracies

                    # Plot the comparison results
                    self.plot_accuracy_diff(class_accuracy_diff, accuracy_diff, dataset_name, pruning_type)
                else:
                    raise Exception('This should never happen.')



if __name__ == "__main__":
    evaluator = ModelEvaluator(base_dir='./Models/')
    evaluator.evaluate_saved_models()
