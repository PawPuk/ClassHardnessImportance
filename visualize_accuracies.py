from collections import defaultdict
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import utils
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
                        pruning_type = path_parts[-2]  # 'Models/pruning_type/dataset_name/...'
                        dataset_name = path_parts[-1]
                        saved_model_paths[(pruning_type, dataset_name)].append(os.path.join(root, file))

        return saved_model_paths

    @staticmethod
    def load_model(model_path, num_classes):
        """
        Load a ResNet-18 model from the given path.

        :param model_path: Path to the saved model.
        :param num_classes: Number of classes in the current dataset.
        :return: Loaded model.
        """
        model = ResNet18LowRes(num_classes).cuda()
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
    def evaluate_model(model, test_loader, num_classes):
        """
        Evaluate the model on the test set and return the accuracy, class-level accuracies, precision, recall, and F1
        score.

        :param model: The trained model to evaluate.
        :param test_loader: DataLoader for the test set.
        :param num_classes: Number of classes in the current dataset.
        :return: (overall accuracy, class accuracies, class precisions, class recalls, class F1 scores).
        """
        class_correct, class_total = np.zeros(num_classes), np.zeros(num_classes)
        total_correct, total_samples = 0, 0
        all_labels, all_predictions = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                for i in range(num_classes):
                    class_mask = (labels == i)
                    class_correct[i] += (predicted[class_mask] == i).sum().item()
                    class_total[i] += class_mask.sum().item()

        overall_accuracy = 100 * total_correct / total_samples
        class_accuracies = 100 * class_correct / class_total  # Per-class accuracies

        # Compute precision, recall, and F1 score for each class
        precisions = precision_score(all_labels, all_predictions, labels=np.arange(num_classes), average=None)
        recalls = recall_score(all_labels, all_predictions, labels=np.arange(num_classes), average=None)
        f1_scores = f1_score(all_labels, all_predictions, labels=np.arange(num_classes), average=None)

        return overall_accuracy, class_accuracies, precisions, recalls, f1_scores

    def evaluate_ensemble(self, model_paths, test_loader, num_classes):
        """
        Evaluate the ensemble of models and compute the average and std for the dataset-level accuracy,
        class-level accuracies, precisions, recalls, and F1 scores.

        :param model_paths: List of paths to saved models.
        :param test_loader: DataLoader for the test set.
        :param num_classes: Number of classes in the current dataset
        :return: (avg_accuracy, std_accuracy, avg_class_accuracies, std_class_accuracies, avg_class_precisions,
                 avg_class_recalls, avg_class_f1_scores).
        """
        accuracies, class_accuracies, class_precisions, class_recalls, class_f1_scores = [], [], [], [], []

        for model_path in model_paths:
            model = self.load_model(model_path, num_classes)
            accuracy, class_acc, precisions, recalls, f1_scores = self.evaluate_model(model, test_loader, num_classes)
            accuracies.append(accuracy)
            class_accuracies.append(class_acc)
            class_precisions.append(precisions)
            class_recalls.append(recalls)
            class_f1_scores.append(f1_scores)

        # Convert to numpy for easy computation
        accuracies = np.array(accuracies)
        class_accuracies = np.array(class_accuracies)
        class_precisions = np.array(class_precisions)
        class_recalls = np.array(class_recalls)
        class_f1_scores = np.array(class_f1_scores)

        # Compute mean and std
        avg_accuracy, std_accuracy = np.mean(accuracies), np.std(accuracies)
        avg_class_accuracies = np.mean(class_accuracies, axis=0)
        std_class_accuracies = np.std(class_accuracies, axis=0)
        avg_class_precisions = np.mean(class_precisions, axis=0)
        std_class_precisions = np.std(class_precisions, axis=0)
        avg_class_recalls = np.mean(class_recalls, axis=0)
        std_class_recalls = np.std(class_recalls, axis=0)
        avg_class_f1_scores = np.mean(class_f1_scores, axis=0)
        std_class_f1_scores = np.std(class_f1_scores, axis=0)

        return (avg_accuracy, std_accuracy, avg_class_accuracies, std_class_accuracies, avg_class_precisions,
                std_class_precisions, avg_class_recalls, std_class_recalls, avg_class_f1_scores, std_class_f1_scores)

    @staticmethod
    def plot_metrics(metric_name, avg_values, std_values, dataset_name, pruning_type, num_classes, avg_overall=None,
                     std_overall=None):
        """
        Plot class-level metrics (accuracy, precision, recall, F1) for the ensemble.
        For accuracy, it also plots the dataset-level overall accuracy as a horizontal line.

        :param metric_name: Name of the metric (e.g., 'Accuracy', 'Precision', 'Recall', 'F1 Score').
        :param avg_values: Array of average metric values per class.
        :param std_values: Array of std values per class.
        :param dataset_name: The name of the dataset.
        :param pruning_type: The pruning type used.
        :param num_classes: Number of classes in the current dataset.
        :param avg_overall: Overall dataset-level average (only applicable for accuracy).
        :param std_overall: Overall dataset-level standard deviation (only applicable for accuracy).
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot candlesticks for class-level metrics
        for i in range(num_classes):
            ax.errorbar(i + 1, avg_values[i], yerr=std_values[i], fmt='o', capsize=5, label=f'Class {i + 1}')

        # Plot dataset-level overall metric as a filled region (for accuracy only)
        if metric_name.lower() == 'accuracy' and avg_overall is not None and std_overall is not None:
            x = np.arange(1, num_classes + 1)
            ax.fill_between(x, avg_overall - std_overall, avg_overall + std_overall, color='gray', alpha=0.3,
                            label=f'{metric_name} Overall')
            ax.plot(x, [avg_overall] * num_classes, color='black', linestyle='--')

        ax.set_xlabel('Class')
        ax.set_ylabel(f'{metric_name} (%)')
        ax.set_title(f'{metric_name} for {dataset_name} with {pruning_type}')
        ax.set_xticks(np.arange(1, num_classes + 1))
        save_dir = os.path.join('Figures/', pruning_type, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f'ensemble_{metric_name.lower()}.pdf')
        plt.savefig(file_name)

    @staticmethod
    def plot_metric_diff(class_metric_diff, overall_metric_diff, dataset_name, pruning_type, metric_name, num_classes):
        """
        Plot the difference in any metric between pruned and baseline models as a bar chart for class-level metrics
        and a horizontal line for overall metric difference (only for accuracy).

        :param class_metric_diff: Array of differences in class-level metrics.
        :param overall_metric_diff: Difference in overall dataset-level metric.
        :param dataset_name: The name of the dataset.
        :param pruning_type: The pruning type used.
        :param metric_name: The metric being plotted (e.g., 'Accuracy', 'Precision', 'Recall', 'F1 Score').
        :param num_classes: Number of classes in the current dataset.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot class-level metric differences as bars
        x = np.arange(num_classes)
        ax.bar(x, class_metric_diff, color='b', alpha=0.7)

        # Plot overall metric difference as a horizontal line (for accuracy only)
        if metric_name.lower() == 'accuracy':
            ax.axhline(y=overall_metric_diff, color='r', linestyle='--',
                       label=f'{metric_name} overall difference: {overall_metric_diff:.2f}%')

        ax.set_xlabel('Class')
        ax.set_ylabel(f'{metric_name} Difference (%)')
        ax.set_title(f'{metric_name} Difference (Pruned vs Baseline) for {dataset_name} with {pruning_type}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {i + 1}' for i in x])
        if metric_name.lower() == 'accuracy':
            ax.legend()

        save_dir = os.path.join('Figures/', pruning_type, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f'ensemble_{metric_name.lower()}_improvements.pdf')
        plt.savefig(file_name)

    def evaluate_saved_models(self):
        """
        Iterate through all saved models grouped by pruning strategy and dataset, load them, and evaluate the ensemble.
        Now evaluates class-level accuracies, precision, recall, and F1 scores, and compares pruned models to baseline.
        """
        saved_model_groups = self.find_saved_model_paths()

        # First, evaluate baseline (pruning_type == 'none') models
        for (pruning_type, dataset_name), model_paths in saved_model_groups.items():
            num_classes = utils.get_config(dataset_name)['num_classes']
            if pruning_type == 'none':
                print(f'\nEvaluating baseline for Dataset: {dataset_name}')
                test_loader = self.load_dataset(dataset_name)
                # Evaluate the ensemble (accuracy, precision, recall, and F1)
                (avg_accuracy, std_accuracy, avg_class_accuracies, std_class_accuracies,
                 avg_class_precisions, std_class_precisions, avg_class_recalls, std_class_recalls,
                 avg_class_f1_scores, std_class_f1_scores) = self.evaluate_ensemble(model_paths, test_loader,
                                                                                    num_classes)

                # Store baseline results
                self.baseline_results[dataset_name] = {
                    'avg_accuracy': avg_accuracy,
                    'avg_class_accuracies': avg_class_accuracies,
                    'avg_class_precisions': avg_class_precisions,
                    'avg_class_recalls': avg_class_recalls,
                    'avg_class_f1_scores': avg_class_f1_scores
                }

                # Print overall ensemble accuracy
                print(f'Average dataset accuracy (baseline): {avg_accuracy:.2f}% (±{std_accuracy:.2f}%)')

                # Plot the results for accuracy, precision, recall, and F1 score
                self.plot_metrics('Accuracy', avg_class_accuracies, std_class_accuracies, dataset_name, pruning_type,
                                  num_classes, avg_accuracy, std_accuracy)
                self.plot_metrics('Precision', avg_class_precisions, std_class_precisions, dataset_name, pruning_type,
                                  num_classes)
                self.plot_metrics('Recall', avg_class_recalls, std_class_recalls, dataset_name, pruning_type,
                                  num_classes)
                self.plot_metrics('F1 Score', avg_class_f1_scores, std_class_f1_scores, dataset_name, pruning_type,
                                  num_classes)

        # Now evaluate the pruned models and compare them to baseline
        for (pruning_type, dataset_name), model_paths in saved_model_groups.items():
            if pruning_type != 'none':  # Evaluate pruned models only
                num_classes = utils.get_config(dataset_name)['num_classes']
                print(f'\nEvaluating ensemble for Dataset: {dataset_name}, Pruning Type: {pruning_type}')

                test_loader = self.load_dataset(dataset_name)
                # Evaluate the ensemble (accuracy, precision, recall, and F1)
                (avg_accuracy, std_accuracy, avg_class_accuracies, std_class_accuracies,
                 avg_class_precisions, std_class_precisions, avg_class_recalls, std_class_recalls,
                 avg_class_f1_scores, std_class_f1_scores) = self.evaluate_ensemble(model_paths, test_loader,
                                                                                    num_classes)

                # Plot the results for the pruned model
                self.plot_metrics('Accuracy', avg_class_accuracies, std_class_accuracies, dataset_name, pruning_type,
                                  num_classes, avg_accuracy, std_accuracy)
                self.plot_metrics('Precision', avg_class_precisions, std_class_precisions, dataset_name, pruning_type,
                                  num_classes)
                self.plot_metrics('Recall', avg_class_recalls, std_class_recalls, dataset_name, pruning_type,
                                  num_classes)
                self.plot_metrics('F1 Score', avg_class_f1_scores, std_class_f1_scores, dataset_name, pruning_type,
                                  num_classes)

                # Compare to baseline results if available
                if dataset_name in self.baseline_results:
                    # Accuracy comparison
                    baseline_accuracy = self.baseline_results[dataset_name]['avg_accuracy']
                    baseline_class_accuracies = self.baseline_results[dataset_name]['avg_class_accuracies']
                    accuracy_diff = avg_accuracy - baseline_accuracy
                    class_accuracy_diff = avg_class_accuracies - baseline_class_accuracies

                    # Precision comparison
                    baseline_class_precisions = self.baseline_results[dataset_name]['avg_class_precisions']
                    class_precision_diff = avg_class_precisions - baseline_class_precisions

                    # Recall comparison
                    baseline_class_recalls = self.baseline_results[dataset_name]['avg_class_recalls']
                    class_recall_diff = avg_class_recalls - baseline_class_recalls

                    # F1 Score comparison
                    baseline_class_f1_scores = self.baseline_results[dataset_name]['avg_class_f1_scores']
                    class_f1_score_diff = avg_class_f1_scores - baseline_class_f1_scores

                    # Plot the comparison results for accuracy, precision, recall, and F1 score
                    self.plot_metric_diff(class_accuracy_diff, accuracy_diff, dataset_name, pruning_type,
                                          'Accuracy', num_classes)
                    self.plot_metric_diff(class_precision_diff, None, dataset_name, pruning_type,
                                          'Precision', num_classes)
                    self.plot_metric_diff(class_recall_diff, None, dataset_name, pruning_type,
                                          'Recall', num_classes)
                    self.plot_metric_diff(class_f1_score_diff, None, dataset_name, pruning_type,
                                          'F1 Score', num_classes)


if __name__ == "__main__":
    evaluator = ModelEvaluator(base_dir='./Models/')
    evaluator.evaluate_saved_models()
