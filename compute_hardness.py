import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from neural_networks import ResNet18LowRes
from utils import get_config


class HardnessCalculator:
    def __init__(self, dataset_name):
        self.seed = 42
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = True  # Set to False for reproducibility
        torch.backends.cudnn.deterministic = True

        self.dataset_name = dataset_name
        config = get_config(dataset_name)
        self.BATCH_SIZE = config['batch_size']
        self.SAVE_EPOCH = config['save_epoch']
        self.MODEL_DIR = config['save_dir']
        self.NUM_CLASSES = config['num_classes']
        self.NUM_MODELS = config['num_models']

        self.training_loader, _, self.training_set_size = self.load_dataset(self.dataset_name)
        self.figure_save_dir = os.path.join('Figures/', self.dataset_name)
        self.results_save_dir = os.path.join('Results/', self.dataset_name)
        os.makedirs(self.figure_save_dir, exist_ok=True)
        os.makedirs(self.results_save_dir, exist_ok=True)

    def load_dataset(self, dataset_name):
        if dataset_name == 'CIFAR10':
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                        transform=train_transform)
        elif dataset_name == 'CIFAR100':
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            training_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                         transform=train_transform)
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        def worker_init_fn(worker_id):
            np.random.seed(self.seed + worker_id)
            random.seed(self.seed + worker_id)

        training_loader = DataLoader(training_set, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=2,
                                     worker_init_fn=worker_init_fn)
        return training_loader, None, len(training_set)

    def create_model(self):
        model = ResNet18LowRes(num_classes=self.NUM_CLASSES)
        return model

    def compute_el2n(self, model, dataloader):
        model.eval()
        el2n_scores = []
        correct, total = 0, 0
        class_correct = {i: 0 for i in range(self.NUM_CLASSES)}
        class_total = {i: 0 for i in range(self.NUM_CLASSES)}

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)

                # Compute EL2N scores
                softmax_outputs = F.softmax(outputs, dim=1)
                one_hot_labels = F.one_hot(labels, num_classes=self.NUM_CLASSES).float()
                l2_errors = torch.norm(softmax_outputs - one_hot_labels, dim=1)
                el2n_scores.extend(l2_errors.cpu().numpy())

                # Compute accuracies
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Class-level accuracies
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

        dataset_accuracy = correct / total
        class_accuracies = {k: class_correct[k] / class_total[k] if class_total[k] > 0 else 0 for k in class_correct}

        return el2n_scores, dataset_accuracy, class_accuracies

    def load_model_and_compute_el2n(self, model_id):
        model = self.create_model().cuda()
        model_path = os.path.join(self.MODEL_DIR, 'none', self.dataset_name,
                                  f'model_{model_id}_epoch_{self.SAVE_EPOCH}.pth')

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            el2n_scores, dataset_accuracy, class_accuracies = self.compute_el2n(model, self.training_loader)
            return el2n_scores, dataset_accuracy, class_accuracies
        else:
            print(f'Model {model_id} not found at epoch {self.SAVE_EPOCH}.')
            return None, None, None

    def collect_el2n_scores(self):
        all_el2n_scores, model_dataset_accuracies, model_class_accuracies = (
            [[] for _ in range(self.training_set_size)],
            [],
            [],
        )
        for model_id in range(self.NUM_MODELS):
            el2n_scores, dataset_accuracy, class_accuracies = self.load_model_and_compute_el2n(model_id)
            if el2n_scores:
                for i in range(self.training_set_size):
                    all_el2n_scores[i].append(el2n_scores[i])
                model_dataset_accuracies.append(dataset_accuracy)
                model_class_accuracies.append(class_accuracies)
        return all_el2n_scores, model_dataset_accuracies, model_class_accuracies

    def group_scores_by_class(self, el2n_scores):
        class_el2n_scores = {i: [] for i in range(self.NUM_CLASSES)}
        labels = []  # Store corresponding labels

        # Since we are not shuffling the data loader, we can directly match scores with their labels
        for i, (_, label) in enumerate(self.training_loader.dataset):
            class_el2n_scores[label].append(el2n_scores[i])
            labels.append(label)

        return class_el2n_scores, labels

    @staticmethod
    def compute_class_statistics(class_el2n_scores):
        class_stats = {}

        for class_id, scores in class_el2n_scores.items():
            scores_array = np.array(scores)
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
        # Unsorted Plot
        class_ids = list(class_stats.keys())
        q1_values = [class_stats[class_id]["q1"] for class_id in class_ids]
        q3_values = [class_stats[class_id]["q3"] for class_id in class_ids]
        min_values = [class_stats[class_id]["min"] for class_id in class_ids]
        max_values = [class_stats[class_id]["max"] for class_id in class_ids]

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(self.NUM_CLASSES):
            ax.plot([i, i], [min_values[i], max_values[i]], color='black')
            ax.plot([i, i], [q1_values[i], q3_values[i]], color='blue', lw=6)

        ax.set_xticks([])
        ax.set_ylabel("EL2N Score (L2 Norm)")
        ax.set_title("Class-Level EL2N Scores Candlestick Plot")

        unsorted_file_name = os.path.join(self.figure_save_dir, 'hardness_distribution.pdf')
        plt.savefig(unsorted_file_name)
        plt.close()

        # Sorted Plot by Q1 for more monotonic appearance
        sorted_class_ids = sorted(class_ids, key=lambda cid: class_stats[cid]["q1"])
        sorted_q1_values = [class_stats[class_id]["q1"] for class_id in sorted_class_ids]
        sorted_q3_values = [class_stats[class_id]["q3"] for class_id in sorted_class_ids]
        sorted_min_values = [class_stats[class_id]["min"] for class_id in sorted_class_ids]
        sorted_max_values = [class_stats[class_id]["max"] for class_id in sorted_class_ids]

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(self.NUM_CLASSES):
            ax.plot([i, i], [sorted_min_values[i], sorted_max_values[i]], color='black')
            ax.plot([i, i], [sorted_q1_values[i], sorted_q3_values[i]], color='blue', lw=6)

        ax.set_xticks([])
        ax.set_ylabel("EL2N Score (L2 Norm)")
        ax.set_title("Sorted Class-Level EL2N Scores Candlestick Plot")

        sorted_file_name = os.path.join(self.figure_save_dir, 'hardness_distribution_sorted.pdf')
        plt.savefig(sorted_file_name)
        plt.close()

    def plot_dataset_level_distribution(self, all_el2n_scores):
        # Compute average and standard deviation across models for each data sample
        avg_scores = [np.mean(scores) for scores in all_el2n_scores]
        std_scores = [np.std(scores) for scores in all_el2n_scores]

        # Sort the average scores and corresponding standard deviations
        sorted_indices = np.argsort(avg_scores)
        sorted_avg_scores = np.array(avg_scores)[sorted_indices]
        sorted_std_scores = np.array(std_scores)[sorted_indices]

        # Plot without standard deviation
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_avg_scores)
        plt.xlabel("Data Sample Index (Sorted)")
        plt.ylabel("Average EL2N Score")
        plt.title("Dataset-Level Hardness Distribution (Without Std)")

        file_name = os.path.join(self.figure_save_dir, f'dataset_level_hardness_distribution_no_std.pdf')
        plt.savefig(file_name)
        plt.close()

        # Plot with standard deviation using fill_between
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_avg_scores, label="Average EL2N Score")
        plt.fill_between(
            range(len(sorted_avg_scores)),
            sorted_avg_scores - sorted_std_scores,
            sorted_avg_scores + sorted_std_scores,
            color="gray",
            alpha=0.2
        )
        plt.xlabel("Data Sample Index (Sorted)")
        plt.ylabel("Average EL2N Score")
        plt.title("Dataset-Level Hardness Distribution (With Std)")

        file_name = os.path.join(self.figure_save_dir, f'dataset_level_hardness_distribution_with_std.pdf')
        plt.savefig(file_name)
        plt.close()

    def plot_class_level_distribution(self, class_el2n_scores):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot without standard deviation
        for class_id, scores in class_el2n_scores.items():
            # Compute average and standard deviation across models for each data sample in this class
            avg_scores = [np.mean(sample_scores) for sample_scores in scores]
            sorted_avg_scores = np.sort(avg_scores)
            ax.plot(sorted_avg_scores)

        ax.set_xlabel("Data Sample Index (Sorted)")
        ax.set_ylabel("Average EL2N Score")
        ax.set_title("Class-Level Hardness Distribution (Without Std)")

        file_name = os.path.join(self.figure_save_dir, f'class_level_hardness_distribution_no_std.pdf')
        plt.savefig(file_name)
        plt.close()

        # Plot with standard deviation using fill_between
        fig, ax = plt.subplots(figsize=(10, 6))

        for class_id, scores in class_el2n_scores.items():
            avg_scores = [np.mean(sample_scores) for sample_scores in scores]
            std_scores = [np.std(sample_scores) for sample_scores in scores]
            sorted_indices = np.argsort(avg_scores)
            sorted_avg_scores = np.array(avg_scores)[sorted_indices]
            sorted_std_scores = np.array(std_scores)[sorted_indices]

            ax.plot(sorted_avg_scores)
            ax.fill_between(
                range(len(sorted_avg_scores)),
                sorted_avg_scores - sorted_std_scores,
                sorted_avg_scores + sorted_std_scores,
                alpha=0.2
            )

        ax.set_xlabel("Data Sample Index (Sorted)")
        ax.set_ylabel("Average EL2N Score")
        ax.set_title("Class-Level Hardness Distribution (With Std)")

        file_name = os.path.join(self.figure_save_dir, f'class_level_hardness_distribution_with_std.pdf')
        plt.savefig(file_name)
        plt.close()

    @staticmethod
    def normalize_el2n_scores(all_el2n_scores):
        # Compute average EL2N score per sample across models
        avg_el2n_scores = [np.mean(scores) for scores in all_el2n_scores]

        # Compute global min and max of averaged scores
        min_score, max_score = min(avg_el2n_scores), max(avg_el2n_scores)

        # Normalize the averaged EL2N scores
        normalized_avg_el2n_scores = [(score - min_score) / (max_score - min_score) for score in avg_el2n_scores]

        return normalized_avg_el2n_scores

    @staticmethod
    def normalize_class_el2n_scores(class_el2n_scores):
        normalized_class_el2n_scores = {}
        for class_id, scores in class_el2n_scores.items():
            # Compute average EL2N score per sample across models within the class
            avg_scores = [np.mean(sample_scores) for sample_scores in scores]

            # Compute min and max for these averaged scores
            min_score, max_score = min(avg_scores), max(avg_scores)

            # Normalize the averaged scores
            normalized_scores = [(score - min_score) / (max_score - min_score) for score in avg_scores]

            normalized_class_el2n_scores[class_id] = normalized_scores
        return normalized_class_el2n_scores

    def plot_pruning_rates(self, normalized_el2n_scores, normalized_class_el2n_scores, labels):
        # Define hardness thresholds
        thresholds = np.linspace(0, 1, 100)

        # Initialize arrays to store pruning percentages
        dataset_pruning_percentages = []
        class_pruning_percentages = {class_id: [] for class_id in range(self.NUM_CLASSES)}

        total_samples = len(normalized_el2n_scores)
        class_sample_counts = {class_id: labels.count(class_id) for class_id in range(self.NUM_CLASSES)}

        for threshold in thresholds:
            # Compute dataset-level pruning percentage
            num_pruned = sum(1 for score in normalized_el2n_scores if score < threshold)
            dataset_pruning_percentage = (num_pruned / total_samples) * 100
            dataset_pruning_percentages.append(dataset_pruning_percentage)

            # Compute class-level pruning percentages
            for class_id in range(self.NUM_CLASSES):
                class_scores = normalized_class_el2n_scores[class_id]
                num_pruned_class = sum(1 for score in class_scores if score < threshold)
                class_pruning_percentage = (num_pruned_class / class_sample_counts[class_id]) * 100
                class_pruning_percentages[class_id].append(class_pruning_percentage)

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
        axes = axes.flatten()

        for class_id in range(self.NUM_CLASSES):
            ax = axes[class_id]
            ax.plot(thresholds, dataset_pruning_percentages)
            ax.plot(thresholds, class_pruning_percentages[class_id])
            ax.set_xlabel('Hardness Threshold')
            ax.set_ylabel('Percentage of Data Pruned')
            ax.set_title(f'Class {class_id}')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        file_name = os.path.join(self.figure_save_dir, 'pruning_rates_vs_hardness.pdf')
        plt.savefig(file_name)
        plt.close()

    def save_el2n_scores(self, el2n_scores):
        with open(os.path.join(self.results_save_dir, 'el2n_scores.pkl'), 'wb') as file:
            pickle.dump(el2n_scores, file)

    def run(self):
        all_el2n_scores, dataset_accuracies, class_accuracies = self.collect_el2n_scores()
        class_el2n_scores, labels = self.group_scores_by_class(all_el2n_scores)
        self.save_el2n_scores((all_el2n_scores, class_el2n_scores, labels, dataset_accuracies, class_accuracies))
        print("Hardness scores computed and saved. Now producing visualization figures.")

        # Normalize EL2N scores after averaging across models
        normalized_el2n_scores = self.normalize_el2n_scores(all_el2n_scores)
        normalized_class_el2n_scores = self.normalize_class_el2n_scores(class_el2n_scores)
        print("Normalized EL2N scores computed. Now producing visualization figures.")

        # Generate and save additional distribution plots
        class_stats = self.compute_class_statistics(class_el2n_scores)
        self.plot_class_level_candlestick(class_stats)
        self.plot_dataset_level_distribution(all_el2n_scores)
        self.plot_class_level_distribution(class_el2n_scores)

        # Plot pruning rates using the normalized scores
        self.plot_pruning_rates(normalized_el2n_scores, normalized_class_el2n_scores, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Hardness of Dataset Samples')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='Specify the dataset name (default: CIFAR10)')
    args = parser.parse_args()

    calculator = HardnessCalculator(args.dataset_name)
    calculator.run()

# TODO: Currently the code works for objective hardness but not for subjective hardness.
