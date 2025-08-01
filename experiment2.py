import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
        self.NUM_TRAINING_SAMPLES = config['num_training_samples']

        self.results_save_dir = os.path.join(ROOT, 'Results/')
        self.figure_save_dir = os.path.join(ROOT, 'Figures/')

    def compute_imbalance_ratio_with_hardness(self, hardness_scores, labels):
        hardnesses_by_class, hardness_of_classes = {class_id: [] for class_id in range(self.NUM_CLASSES)}, {}
        estimates = np.mean(np.array(hardness_scores), axis=0)

        for i, label in enumerate(labels):
            hardnesses_by_class[label].append(estimates[i])

        for class_id in range(self.NUM_CLASSES):
            if self.hardness_estimator in ['AUM', 'Confidence', 'iAUM', 'iConfidence']:
                hardness_of_classes[class_id] = 1 / np.mean(hardnesses_by_class[class_id])
            else:
                hardness_of_classes[class_id] = np.mean(hardnesses_by_class[class_id])
        ratios = {class_id: class_hardness / sum(hardness_of_classes.values())
                  for class_id, class_hardness in hardness_of_classes.items()}
        samples_per_class = np.array([int(round(ratio * sum(self.NUM_TRAINING_SAMPLES)))
                                      for class_id, ratio in ratios.items()])

        return samples_per_class

    def investigate_resampling_ratios(self, hardness_estimates, labels, thresholds):
        per_class_counts, pearson_scores, spearman_scores, ideal_ratios, window_size = {}, {}, {}, {}, 5
        for estimator_name in hardness_estimates.keys():
            if estimator_name == 'probs':
                continue
            estimates = np.mean(np.array(hardness_estimates[estimator_name]), axis=0)
            sorted_indices = np.argsort(estimates)
            num_samples = len(estimates)
            per_class_counts[estimator_name] = []
            pearson_scores[estimator_name], spearman_scores[estimator_name] = [], []
            for t in thresholds:
                retain_count = int((100 - t) / 100 * num_samples)
                retained_indices = sorted_indices[:retain_count]
                counts = np.zeros(self.NUM_CLASSES, dtype=int)
                for idx in retained_indices:
                    cls = labels[idx]
                    counts[cls] += 1
                per_class_counts[estimator_name].append(counts)
            for i in range(len(per_class_counts[estimator_name]) - 1):
                x = per_class_counts[estimator_name][i]
                y = per_class_counts[estimator_name][i + 1]
                pearson_scores[estimator_name].append(np.corrcoef(x, y)[0, 1])
                spearman_scores[estimator_name].append(pd.Series(x).corr(pd.Series(y), method='spearman'))
            pearson_avg = pd.Series(pearson_scores[estimator_name]).rolling(window=window_size, min_periods=1).mean()
            spearman_avg = pd.Series(spearman_scores[estimator_name]).rolling(window=window_size, min_periods=1).mean()
            combined_avg = (pearson_avg + spearman_avg) / 2
            best_combined_idx = combined_avg.idxmax()
            ideal_ratios[estimator_name] = per_class_counts[estimator_name][best_combined_idx]
            # ideal_ratios[estimator_name] = per_class_counts[estimator_name][25]  # sanity check
            print(f'{best_combined_idx} is the best ratio for {estimator_name}.')
        # Hardness-Based Resampling Ratio computation
        # ideal_ratios['HBRR'] = self.compute_imbalance_ratio_with_hardness(hardness_estimates['AUM'], labels)
        ideal_ratios['HBRR'] = self.compute_imbalance_ratio_with_hardness(hardness_estimates['Confidence'], labels)  # sanity check
        return per_class_counts, pearson_scores, spearman_scores, ideal_ratios

    def measure_stability_of_resampling_ratios(self, pearson_scores, spearman_scores, ideal_ratios, hardness_estimates,
                                               thresholds):
        for estimator_name in hardness_estimates.keys():
            if estimator_name == 'probs':
                continue
            plt.figure(figsize=(8, 5))
            plt.plot(thresholds[:-1], pearson_scores[estimator_name], label='Pearson', color='blue')
            plt.plot(thresholds[:-1], spearman_scores[estimator_name], label='Spearman', color='orange')
            plt.xlabel("Pruning Threshold (%)")
            plt.ylabel("Inter-Class Distribution Correlation")
            plt.title(f"Class Retention Correlation Across Pruning Thresholds ({estimator_name})")
            plt.legend()
            plt.grid(True)
            fig_path = os.path.join(self.figure_save_dir, f"unclean{self.dataset_name}",
                                    f"correlation_threshold_stability_{estimator_name}.pdf")
            plt.savefig(fig_path)
            plt.close()

        ideal_ratios_df = pd.DataFrame(ideal_ratios)
        correlation_matrix = ideal_ratios_df.corr(method='pearson')  # or 'spearman'
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Pearson Correlation Matrix of Ideal Resampling Ratios")
        fig.savefig(os.path.join(self.figure_save_dir, f"unclean{self.dataset_name}",
                                 "resampling_ratio_pearson_correlation_matrix.pdf"))
        plt.close()
        correlation_matrix = ideal_ratios_df.corr(method='spearman')  # or 'spearman'
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Spearman Correlation Matrix of Ideal Resampling Ratios")
        fig.savefig(os.path.join(self.figure_save_dir, f"unclean{self.dataset_name}",
                                 "resampling_ratio_spearman_correlation_matrix.pdf"))
        plt.close()

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

        hardness_estimates = load_hardness_estimates('unclean', self.dataset_name)
        thresholds = np.arange(0, 100, 1)
        per_class_counts, pearson_scores, spearman_scores, ideal_ratios = self.investigate_resampling_ratios(
            hardness_estimates, labels, thresholds)
        self.measure_stability_of_resampling_ratios(pearson_scores, spearman_scores, ideal_ratios, hardness_estimates,
                                                    thresholds)
        if self.pruning_type == 'easy':
            if self.hardness_estimator in ['DataIQ', 'iDataIQ', 'Forgetting', 'Loss', 'iLoss', 'EL2N']:
                pruned_dataset = self.prune_dataset(labels, training_loader,
                                                    hardness_estimates[self.hardness_estimator], True)
            elif self.hardness_estimator in ['Confidence', 'iConfidence', 'AUM', 'iAUM']:
                pruned_dataset = self.prune_dataset(labels, training_loader,
                                                    hardness_estimates[self.hardness_estimator], False)
            else:
                raise ValueError(f'{self.hardness_estimator} is not a supported hardness estimator.')
        else:
            imbalance_ratio = per_class_counts[self.hardness_estimator][self.pruning_rate]
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
