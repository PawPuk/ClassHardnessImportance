import argparse
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_networks import ResNet18LowRes
from data import load_dataset
from config import get_config, ROOT


class PerformanceVisualizer:
    def __init__(self, dataset_name, pruning_type):
        self.dataset_name = dataset_name
        self.pruning_type = pruning_type

        config = get_config(args.dataset_name)
        self.num_classes = config['num_classes']
        self.num_epochs = config['num_epochs']
        self.num_training_samples = config['num_training_samples']
        self.num_test_samples = config['num_test_samples']
        self.num_models = config['robust_ensemble_size']

        self.figure_save_dir = os.path.join(ROOT, 'Figures/', args.dataset_name)
        self.results_dir = os.path.join(ROOT, "Results/", args.dataset_name)
        os.makedirs(self.figure_save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def load_models(self) -> Dict[str, Dict[int, List[dict]]]:
        models_dir = os.path.join(ROOT, "Models/")
        models_by_rate = {f'{self.pruning_type}_clp': {}, f'{self.pruning_type}_dlp': {}}

        for pruning_strategy in [f'{self.pruning_type}_clp', f'{self.pruning_type}_dlp']:
            # Walk through each folder in the Models directory
            for root, dirs, files in os.walk(models_dir):
                # Ensure the dataset name matches exactly (avoid partial matches like "cifar10" in "cifar100")
                if f"{pruning_strategy}" in root and os.path.basename(root) == f'unclean{self.dataset_name}':
                    pruning_rate = int(root.split(pruning_strategy)[1].split("/")[0])
                    models_by_rate[pruning_strategy].setdefault(pruning_rate, [])
                    for file in files:
                        if file.endswith(".pth") and "_epoch_200" in file:
                            model_path = os.path.join(root, file)
                            model_state = torch.load(model_path)
                            models_by_rate[pruning_strategy][pruning_rate].append(model_state)
            if len(models_by_rate[pruning_strategy][pruning_rate]) < self.num_models:
                self.num_models = len(models_by_rate[pruning_strategy][pruning_rate])

            # Load models trained on the full dataset (no pruning)
            full_dataset_dir = os.path.join(models_dir, "none", f'unclean{self.dataset_name}')
            if os.path.exists(full_dataset_dir):
                models_by_rate[pruning_strategy][0] = []  # Use `0` to represent models without pruning
                for file in os.listdir(full_dataset_dir):
                    if file.endswith(".pth") and "_epoch_200" in file:
                        model_path = os.path.join(full_dataset_dir, file)
                        model_state = torch.load(model_path)
                        models_by_rate[pruning_strategy][0].append(model_state)

            print(f"Models loaded by pruning rate for {pruning_strategy} on {self.dataset_name}")

        return models_by_rate

    def compute_pruned_percentages(self, models_by_rate: Dict[str, Dict[int, List[dict]]]) -> Dict[str, Dict[
                                                                                              int, List[float]]]:
        """Computes the percentage of samples that were pruned per class for each of the pruning rate and pruning
        strategy."""
        pruned_percentages = {}
        for pruning_strategy in [f'{self.pruning_type}_clp', f'{self.pruning_type}_dlp']:
            pruning_rates = models_by_rate[pruning_strategy].keys()
            pruned_percentages[pruning_strategy] = {pruning_rate: [] for pruning_rate in pruning_rates}

            # Iterate over each pruning rate in models_by_rate
            for pruning_rate in pruning_rates:
                if pruning_rate != 0:
                    pkl_path = os.path.join(ROOT, "Results", pruning_strategy + str(pruning_rate), self.dataset_name,
                                            f"class_level_sample_counts.pkl")
                    with open(pkl_path, "rb") as file:
                        class_level_sample_counts = pickle.load(file)
                    pkey = 'clp' if 'clp' in pruning_strategy else 'dlp'
                    remaining_data_count = class_level_sample_counts[pkey][pruning_rate]
                    for c in range(self.num_classes):
                        pruned_percentage = 100.0 * (self.num_training_samples[c] - remaining_data_count[c]) / \
                                            self.num_training_samples[c]
                        pruned_percentages[pruning_strategy][pruning_rate].append(pruned_percentage)
                else:
                    pruned_percentages[pruning_strategy][0] = [0.0 for _ in range(self.num_classes)]

        return pruned_percentages

    def plot_pruned_percentages(self, pruned_percentages: Dict[int, List[float]]):
        plt.figure(figsize=(10, 6))

        pruning_rates = sorted(pruned_percentages.keys())
        for class_idx in range(self.num_classes):
            class_pruning_percentages = [pruned_percentages[pruning_rate][class_idx] for pruning_rate in pruning_rates]
            plt.plot(pruning_rates, class_pruning_percentages, marker='o')

        plt.xlabel("Percentage of samples removed from the dataset (DLP rate)")
        plt.ylabel("Class-Level Pruning Percentage (%)")
        plt.grid(True)
        plt.savefig(os.path.join(self.figure_save_dir, f'class_vs_dataset_pruning_values.pdf'))

    def evaluate_block(self, ensemble: List[dict], test_loader: DataLoader, class_index: int, pruning_rate: int,
                       results: Dict[str, Dict[str, Dict[int, Dict[int, Dict]]]], pruning_strategy: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for model_idx, model_state in enumerate(ensemble):
            if model_idx == self.num_models:
                continue
            Tp, Fp, Fn, Tn = 0, 0, 0, 0
            model = ResNet18LowRes(self.num_classes)
            model.load_state_dict(model_state)
            model = model.to(device)
            model.eval()

            with torch.no_grad():
                for images, labels, _ in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)

                    for pred, label in zip(predicted, labels):
                        if label == class_index:
                            if pred.item() == class_index:
                                Tp += 1
                            else:
                                Fn += 1
                        else:
                            if pred.item() == class_index:
                                Fp += 1
                            else:
                                Tn += 1

            results[pruning_strategy]['Tp'][pruning_rate][class_index][model_idx] = Tp
            results[pruning_strategy]['Fp'][pruning_rate][class_index][model_idx] = Fp
            results[pruning_strategy]['Fn'][pruning_rate][class_index][model_idx] = Fn
            results[pruning_strategy]['Tn'][pruning_rate][class_index][model_idx] = Tn

    def evaluate_ensemble(self, models_by_rate: Dict[str, Dict[int, List[dict]]], test_loader: DataLoader,
                          results: Dict[str, Dict[str, Dict]]):
        for pruning_strategy in [f'{self.pruning_type}_clp', f'{self.pruning_type}_dlp']:
            for pruning_rate, ensemble in tqdm(models_by_rate[pruning_strategy].items(),
                                               desc=f'Iterating over pruning rates ({pruning_strategy}).'):
                for metric_name in ['Tp', 'Fp', 'Fn', 'Tn']:
                    results[pruning_strategy][metric_name][pruning_rate] = {class_id: {}
                                                                            for class_id in range(self.num_classes)}
                for class_index in tqdm(range(self.num_classes), desc='Iterating over classes'):
                    self.evaluate_block(ensemble, test_loader, class_index, pruning_rate, results, pruning_strategy)

    def save_file(self, filename: str, data: Dict[str, Dict[str, Dict[int, Dict[int, Dict[int, int]]]]]):
        save_location = os.path.join(self.results_dir, filename)
        with open(save_location, "wb") as file:
            pickle.dump(data, file)

    def compute_and_plot_recall_correlations_across_ensemble_sizes(self, results):
        for pruning_strategy in [f'{self.pruning_type}_clp', f'{self.pruning_type}_dlp']:
            pruning_rates = sorted(results[pruning_strategy]['Recall'].keys())
            pearson_matrix, spearman_matrix = [], []
            pearson_pval_matrix, spearman_pval_matrix = [], []
            for pruning_rate in sorted(results[pruning_strategy]['Recall'].keys()):
                pearson_row, spearman_row = [], []
                pearson_pval_row, spearman_pval_row = [], []
                for i in range(1, self.num_models):
                    recall_i, recall_ip1 = [], []
                    for class_id in range(self.num_classes):
                        # Ensemble of size i
                        recall_vec_i = np.mean([
                            results[pruning_strategy]['Recall'][pruning_rate][class_id][model_idx]
                            for model_idx in range(i)
                        ])
                        # Ensemble of size i+1
                        recall_vec_ip1 = np.mean([
                            results[pruning_strategy]['Recall'][pruning_rate][class_id][model_idx]
                            for model_idx in range(i + 1)
                        ])
                        recall_i.append(recall_vec_i)
                        recall_ip1.append(recall_vec_ip1)
                    pearson_corr, pearson_pval = pearsonr(recall_i, recall_ip1)
                    spearman_corr, spearman_pval = spearmanr(recall_i, recall_ip1)
                    pearson_row.append(pearson_corr)
                    spearman_row.append(spearman_corr)
                    pearson_pval_row.append(pearson_pval)
                    spearman_pval_row.append(spearman_pval)
                pearson_matrix.append(pearson_row)
                spearman_matrix.append(spearman_row)
                pearson_pval_matrix.append(pearson_pval_row)
                spearman_pval_matrix.append(spearman_pval_row)
            pearson_matrix = np.array(pearson_matrix)
            spearman_matrix = np.array(spearman_matrix)
            pearson_pval_matrix = np.array(pearson_pval_matrix)
            spearman_pval_matrix = np.array(spearman_pval_matrix)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
            x_labels = list(range(1, self.num_models))
            y_labels = [f"{r}%" for r in pruning_rates]
            heatmaps = [
                (pearson_matrix, "Pearson Correlation", axes[0, 0]),
                (spearman_matrix, "Spearman Correlation", axes[0, 1]),
                (pearson_pval_matrix, "Pearson p-value", axes[1, 0]),
                (spearman_pval_matrix, "Spearman p-value", axes[1, 1]),
            ]
            for mat, title, ax in heatmaps:
                sns.heatmap(
                    mat, xticklabels=x_labels, yticklabels=y_labels, annot=True, fmt=".2f",
                    cmap="YlGnBu" if "Correlation" in title else "OrRd", ax=ax,
                    cbar=True, vmin=0.0, vmax=1.0
                )
                ax.set_title(title)
            for ax in axes[1]:  # Label only bottom row
                ax.set_xlabel("Ensemble Size i (vs i+1)")
            for ax in axes[:, 0]:  # Label only left column
                ax.set_ylabel("Pruning Rate")
            fig.suptitle(f"Recall Correlation & p-values Across Ensemble Sizes ({pruning_strategy})", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(os.path.join(self.figure_save_dir, f"recall_corr_pval_heatmap_{pruning_strategy}.pdf"))
            plt.close()

    def plot_class_level_results(self, results: Dict[str, Dict[str, Dict[int, Dict[int, Dict[int, int]]]]],
                                 pruned_percentages: Dict[str, Dict[int, List[float]]]):
        pruning_rates = sorted(results[f'{self.pruning_type}_clp']['Average Accuracy'].keys())

        for metric_name in results[f'{self.pruning_type}_clp'].keys():
            fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey='all')
            fig.suptitle(f"Class-Level {metric_name} Across Pruning Rates", fontsize=16)
            axes = axes.flatten()

            for class_id in range(self.num_classes):
                avg_metric_values_clp = np.array([np.mean([results[f'{self.pruning_type}_clp'][metric_name][p][class_id]
                                                           [model_idx] for model_idx in range(self.num_models)])
                                                  for p in pruning_rates])
                std_metric_values_clp = np.array([np.std([results[f'{self.pruning_type}_clp'][metric_name][p][class_id]
                                                          [model_idx] for model_idx in range(self.num_models)])
                                                  for p in pruning_rates])

                avg_metric_values_dlp = np.array([np.mean([results[f'{self.pruning_type}_dlp'][metric_name][p][class_id]
                                                           [model_idx] for model_idx in range(self.num_models)])
                                                  for p in pruning_rates])
                std_metric_values_dlp = np.array([np.std([results[f'{self.pruning_type}_dlp'][metric_name][p][class_id]
                                                          [model_idx] for model_idx in range(self.num_models)])
                                                  for p in pruning_rates])
                if metric_name == 'Recall':
                    print(f'Class {class_id}:\n\t{avg_metric_values_clp}\n\t{avg_metric_values_dlp}')

                pruning_clp = [pruned_percentages[f'{self.pruning_type}_clp'][p][class_id] for p in pruning_rates]
                pruning_dlp = [pruned_percentages[f'{self.pruning_type}_dlp'][p][class_id] for p in pruning_rates]

                # Plot CLP (Blue Line)
                ax = axes[class_id]
                ax.plot(pruning_clp, avg_metric_values_clp, marker='o', linestyle='-', color='blue', label='clp')
                ax.fill_between(pruning_clp, avg_metric_values_clp - std_metric_values_clp,
                                avg_metric_values_clp + std_metric_values_clp, color='blue', alpha=0.2)
                ax.set_xlabel("Pruning % (clp)", color='blue')
                ax.set_xlim(0, 100)
                ax.tick_params(axis='x', labelcolor='blue')

                # Plot DLP (Red Line)
                ax2 = ax.twiny()
                ax2.plot(pruning_dlp, avg_metric_values_dlp, marker='s', linestyle='--', color='red', label='dlp')
                ax.fill_between(pruning_dlp, avg_metric_values_dlp - std_metric_values_dlp,
                                avg_metric_values_dlp + std_metric_values_dlp, color='red', alpha=0.2)
                ax2.set_xlabel("Pruning % (dlp)", color='red')
                ax2.set_xlim(0, 100)  # Fix x-axis range for DLP
                ax2.tick_params(axis='x', labelcolor='red')

                ax.set_ylabel(f"{metric_name}")
                ax.set_title(f"Class {class_id}")
                ax.grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
            plt.savefig(os.path.join(self.figure_save_dir, f'Class_Level_{metric_name}.pdf'))
            plt.close()

    def compare_clp_with_dlp(self, results: Dict[str, Dict[str, Dict[int, Dict[int, Dict[int, int]]]]]):
        pruning_rates = sorted(results[f'{self.pruning_type}_clp']['Average Accuracy'].keys())

        for metric_name in results[f'{self.pruning_type}_clp'].keys():
            avg_metric_clp = np.array([np.mean([results[f'{self.pruning_type}_clp'][metric_name][p][class_id][model_idx]
                                                for model_idx in range(self.num_models)
                                                for class_id in range(self.num_classes)])
                                       for p in pruning_rates])
            std_metric_clp = np.array([np.std([results[f'{self.pruning_type}_clp'][metric_name][p][class_id][model_idx]
                                               for model_idx in range(self.num_models)
                                               for class_id in range(self.num_classes)])
                                       for p in pruning_rates])

            avg_metric_dlp = np.array([np.mean([results[f'{self.pruning_type}_dlp'][metric_name][p][class_id][model_idx]
                                                for model_idx in range(self.num_models)
                                                for class_id in range(self.num_classes)])
                                       for p in pruning_rates])
            std_metric_dlp = np.array([np.std([results[f'{self.pruning_type}_dlp'][metric_name][p][class_id][model_idx]
                                               for model_idx in range(self.num_models)
                                               for class_id in range(self.num_classes)])
                                       for p in pruning_rates])

            if metric_name == 'Recall':
                print(avg_metric_clp)
                print(avg_metric_dlp)

            plt.figure(figsize=(8, 6))

            # Plot CLP (Blue Line)
            plt.plot(pruning_rates, avg_metric_clp, marker='o', linestyle='-', color='blue', label='CLP')
            plt.fill_between(pruning_rates, avg_metric_clp - std_metric_clp, avg_metric_clp + std_metric_clp,
                             color='blue', alpha=0.2)
            # Plot DLP (Red Line)
            plt.plot(pruning_rates, avg_metric_dlp, marker='s', linestyle='--', color='red', label='DLP')
            plt.fill_between(pruning_rates, avg_metric_dlp - std_metric_dlp, avg_metric_dlp + std_metric_dlp,
                             color='red', alpha=0.2)  # Shaded region for std

            plt.xlabel("Percentage of samples removed from the dataset", fontsize=12)
            plt.ylabel(f"Average {metric_name}", fontsize=12)
            plt.xlim(0, 100)  # Ensure pruning percentage is between 0 and 100
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.savefig(os.path.join(self.figure_save_dir, f'{metric_name}_clp_vs_dlp.pdf'))

    def main(self):
        models = self.load_models()
        pruned_percentages = self.compute_pruned_percentages(models)
        if self.num_classes == 10:
            self.plot_pruned_percentages(pruned_percentages[f'{self.pruning_type}_dlp'])

        # We only use test_loader so values of remove_noise, shuffle, and apply_augmentation doesn't matter below.
        _, _, test_loader, _ = load_dataset(self.dataset_name, False, False, True)

        # Evaluate ensemble performance
        if os.path.exists(os.path.join(self.results_dir, "ensemble_results.pkl")):
            print('Loading pre-computed ensemble results.')
            with open(os.path.join(self.results_dir, "ensemble_results.pkl"), 'rb') as f:
                results = pickle.load(f)
        else:
            results = {}
        if f'{self.pruning_type}_clp' not in results.keys():
            for pruning_strategy in [f'{self.pruning_type}_clp', f'{self.pruning_type}_dlp']:
                results[pruning_strategy] = {}
                for metric_name in ['Tp', 'Fn', 'Fp', 'Tn']:
                    results[pruning_strategy][metric_name] = {}

            self.evaluate_ensemble(models, test_loader, results)
            self.save_file("ensemble_results.pkl", results)

        for pruning_strategy in [f'{self.pruning_type}_clp', f'{self.pruning_type}_dlp']:
            for metric_name in ['F1', 'MCC', 'Average Accuracy', 'Precision', 'Recall']:
                results[pruning_strategy][metric_name] = {}
            for pruning_rate in results[f'{self.pruning_type}_clp']['Tp'].keys():
                for metric_name in ['F1', 'MCC', 'Average Accuracy', 'Precision', 'Recall']:
                    results[pruning_strategy][metric_name][pruning_rate] = {}
                for class_id in range(self.num_classes):
                    for metric_name in ['F1', 'MCC', 'Average Accuracy', 'Precision', 'Recall']:
                        results[pruning_strategy][metric_name][pruning_rate][class_id] = {}
                    for model_idx in range(self.num_models):
                        Tp = results[pruning_strategy]['Tp'][pruning_rate][class_id][model_idx]
                        Fp = results[pruning_strategy]['Fp'][pruning_rate][class_id][model_idx]
                        Fn = results[pruning_strategy]['Fn'][pruning_rate][class_id][model_idx]
                        Tn = results[pruning_strategy]['Tn'][pruning_rate][class_id][model_idx]

                        precision = Tp / (Tp + Fp) if (Tp + Fp) > 0 else 0.0
                        recall = Tp / (Tp + Fn) if (Tp + Fn) > 0 else 0.0
                        accuracy = Tp + Tn / sum(self.num_test_samples)
                        F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                        MCC_numerator = Tp * Tn - Fp * Fn
                        MCC_denominator = ((Tp + Fp) * (Tp + Fn) * (Tn + Fp) * (Tn + Fn)) ** 0.5
                        MCC = 0.0 if MCC_denominator == 0 else MCC_numerator / MCC_denominator

                        results[pruning_strategy]['F1'][pruning_rate][class_id][model_idx] = F1
                        results[pruning_strategy]['MCC'][pruning_rate][class_id][model_idx] = MCC
                        results[pruning_strategy]['Average Accuracy'][pruning_rate][class_id][model_idx] = accuracy
                        results[pruning_strategy]['Precision'][pruning_rate][class_id][model_idx] = precision
                        results[pruning_strategy]['Recall'][pruning_rate][class_id][model_idx] = recall

        self.compute_and_plot_recall_correlations_across_ensemble_sizes(results)
        if self.num_classes == 10:
            self.plot_class_level_results(results, pruned_percentages)
        self.compare_clp_with_dlp(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., 'CIFAR10')")
    parser.add_argument("--pruning_type", choices=['easy', 'random'])
    args = parser.parse_args()
    PerformanceVisualizer(args.dataset_name, args.pruning_type).main()

# TODO: Plot stability of recall gap as a function of ensemble size and pruning rate (1 plot per pruning rate)
# TODO: Repeat the sum of absolute differences in class-level recall values (1 plot per pruning rate).
