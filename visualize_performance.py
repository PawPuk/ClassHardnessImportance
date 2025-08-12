import argparse
import os
import pickle
from statistics import mean, stdev
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_networks import ResNet18LowRes
from data import load_dataset
from config import get_config, ROOT
from utils import load_results


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
        self.results_dir = os.path.join(ROOT, "Results/")
        os.makedirs(self.figure_save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def load_models(self) -> Dict[str, Dict[int, List[str]]]:
        models_dir = os.path.join(ROOT, "Models/")
        models_by_rate, pruning_rate = {f'{self.pruning_type}_clp': {}, f'{self.pruning_type}_dlp': {}}, None

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
                            models_by_rate[pruning_strategy][pruning_rate].append(model_path)
            if len(models_by_rate[pruning_strategy][pruning_rate]) < self.num_models:
                self.num_models = len(models_by_rate[pruning_strategy][pruning_rate])

            # Load models trained on the full dataset (no pruning)
            full_dataset_dir = os.path.join(models_dir, "none", f'unclean{self.dataset_name}')
            if os.path.exists(full_dataset_dir):
                models_by_rate[pruning_strategy][0] = []  # Use `0` to represent models without pruning
                for file in os.listdir(full_dataset_dir):
                    if file.endswith(".pth") and "_epoch_200" in file:
                        model_path = os.path.join(full_dataset_dir, file)
                        models_by_rate[pruning_strategy][0].append(model_path)

            print(f"Models loaded by pruning rate for {pruning_strategy} on {self.dataset_name}")

        return models_by_rate

    def compute_pruned_percentages(self, models_by_rate: Dict[str, Dict[int, List[str]]]) -> Dict[str, Dict[
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

    def evaluate_block(self, ensemble: List[str], test_loader: DataLoader, class_index: int, pruning_rate: int,
                       results: Dict[str, Dict[str, Dict[int, Dict[int, Dict]]]], pruning_strategy: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for model_idx, model_path in enumerate(ensemble):
            model_state = torch.load(model_path)
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

    def evaluate_ensemble(self, models_by_rate: Dict[str, Dict[int, List[str]]], test_loader: DataLoader,
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
        save_location = os.path.join(self.results_dir, self.dataset_name, filename)
        os.makedirs(save_location, exist_ok=True)
        with open(save_location, "wb") as file:
            pickle.dump(data, file)

    def plot_classwise_recall_vs_pruning(self, results):
        clp_key = f"{self.pruning_type}_clp"
        dlp_key = f"{self.pruning_type}_dlp"

        pruning_rates = list(results[clp_key]['Recall'].keys())
        zero_rate = 0.0 if 0.0 in pruning_rates else 0
        pruning_rate_no_zero = [pr for pr in pruning_rates if pr != zero_rate]

        fig, axes = plt.subplots(nrows=1, ncols=len(pruning_rate_no_zero), figsize=(4 * len(pruning_rate_no_zero), 8),
                                 sharey=True)
        for ax, pruning_rate in zip(axes, pruning_rate_no_zero):
            clp_recalls = [np.mean(list(results[clp_key]['Recall'][pruning_rate][cls].values()))
                           for cls in range(self.num_classes)]
            dlp_recalls = [np.mean(list(results[dlp_key]['Recall'][pruning_rate][cls].values()))
                           for cls in range(self.num_classes)]
            sorted_classes = np.argsort(clp_recalls)[::-1]
            clp_sorted = np.array(clp_recalls)[sorted_classes]
            dlp_sorted = np.array(dlp_recalls)[sorted_classes]

            ax.plot(clp_sorted, label=f'CLP (rate={pruning_rate})', color='blue')
            ax.plot(dlp_sorted, label=f'DLP (rate={pruning_rate})', color='red')

            ax.set_title(f'Class-wise Recall @ pruning_rate={pruning_rate}')
            ax.set_ylabel('Recall')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        axes[-1].set_xlabel('Class (sorted by baseline recall)')
        plt.tight_layout()
        filename = f"class_level_recall_changes_investigation.pdf"
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_save_dir, filename))

    def compute_fairness_metrics(self, results, samples_per_class):
        fairness_results = {}
        for pruning_strategy in [f'{self.pruning_type}_clp', f'{self.pruning_type}_dlp']:
            fairness_results[pruning_strategy] = {
                'max_min_gap': {},  # strategy -> pruning_rate -> ensemble_size -> float
                'std_dev': {},
                'cv': {},
                'quant_diff': {},
                'hard_easy': {},
                'avg_change': {}
            }
            for pruning_rate in sorted(results[pruning_strategy]['Recall'].keys()):
                for fairness_metric in 'max_min_gap', 'std_dev', 'cv', 'quant_diff', 'hard_easy', 'avg_change':
                    fairness_results[pruning_strategy][fairness_metric][pruning_rate] = {}

                for ensemble_size in range(1, self.num_models + 1):
                    recall_values = []
                    for class_id in range(self.num_classes):
                        # Mean recall over ensemble_size models for a given class
                        mean_recall = np.mean([
                            results[pruning_strategy]['Recall'][pruning_rate][class_id][model_idx]
                            for model_idx in range(ensemble_size)
                        ])
                        recall_values.append(mean_recall)

                    max_min_gap = max(recall_values) - min(recall_values)
                    std_dev = np.std(recall_values)
                    cv = stdev(recall_values) / mean(recall_values)
                    quant_diff = np.percentile(recall_values, 90) - np.percentile(recall_values, 10)
                    hard_class_recalls = [recall_values[cls] for cls in samples_per_class
                                         if samples_per_class[cls] > np.mean(list(samples_per_class.values()))]
                    easy_class_recalls = [recall_values[cls] for cls in samples_per_class
                                         if samples_per_class[cls] <= np.mean(list(samples_per_class.values()))]
                    hard_easy = abs(mean(hard_class_recalls) - mean(easy_class_recalls))
                    base_values = [mean([results[pruning_strategy]['Recall'][0][class_id][model_idx]
                                        for model_idx in range(ensemble_size)])
                                   for class_id in range(self.num_classes)]
                    avg_change = mean(base_values) - mean(recall_values)

                    fairness_results[pruning_strategy]['max_min_gap'][pruning_rate][ensemble_size] = max_min_gap
                    fairness_results[pruning_strategy]['std_dev'][pruning_rate][ensemble_size] = std_dev
                    fairness_results[pruning_strategy]['cv'][pruning_rate][ensemble_size] = cv
                    fairness_results[pruning_strategy]['quant_diff'][pruning_rate][ensemble_size] = quant_diff
                    fairness_results[pruning_strategy]['hard_easy'][pruning_rate][ensemble_size] = hard_easy
                    fairness_results[pruning_strategy]['avg_change'][pruning_rate][ensemble_size] = avg_change

        return fairness_results


    def generate_fairness_table(self, fairness_results, save_path):
        rows = []
        for pruning_strategy, metrics_dict in fairness_results.items():
            for pruning_rate in metrics_dict['max_min_gap']:
                row = {
                    'Pruning Strategy': pruning_strategy,
                    'Pruning Rate': pruning_rate,
                    'max_min_gap': metrics_dict['max_min_gap'][pruning_rate][self.num_models],
                    'std_dev': metrics_dict['std_dev'][pruning_rate][self.num_models],
                    'cv': metrics_dict['cv'][pruning_rate][self.num_models],
                    'quant_diff': metrics_dict['quant_diff'][pruning_rate][self.num_models],
                    'hard_easy': metrics_dict['hard_easy'][pruning_rate][self.num_models],
                    'avg_change': metrics_dict['avg_change'][pruning_rate][self.num_models]
                }
                rows.append(row)
        df = pd.DataFrame(rows)

        def quantize(s):
            return [f'{v:.4f}' for v in s]

        styled_df = df.copy()
        for col in ['max_min_gap', 'std_dev', 'cv', 'quant_diff', 'hard_easy', 'avg_change']:
            styled_df[col] = quantize(df[col])
        df.to_csv(os.path.join(save_path, f"pruning_fairness_results.csv"), index=False)
        latex_str = styled_df.to_latex(index=False, escape=False)
        with open(os.path.join(save_path, f"pruning_fairness_results.tex"), "w") as f:
            f.write(latex_str)

    def plot_fairness_stability(self, fairness_results):
        for pruning_strategy in [f'{self.pruning_type}_clp', f'{self.pruning_type}_dlp']:
            for fairness_type in ['max_min_gap', 'std_dev', 'cv', 'quant_diff', 'hard_easy', 'avg_change']:
                # Extract data into a 2D matrix: rows = pruning rates, cols = ensemble sizes
                pruning_rates = sorted(fairness_results[pruning_strategy][fairness_type].keys())
                ensemble_sizes = sorted(next(iter(fairness_results[pruning_strategy][fairness_type].values())).keys())

                heatmap_data = np.array([
                    [fairness_results[pruning_strategy][fairness_type][rate][size]
                     for size in ensemble_sizes]
                    for rate in pruning_rates
                ])

                plt.figure(figsize=(10, 6))
                sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis",
                            xticklabels=ensemble_sizes, yticklabels=[f"{r}%" for r in pruning_rates])
                plt.title(f"{fairness_type.replace('_', ' ').title()} Heatmap ({pruning_strategy})")
                plt.xlabel("Ensemble Size")
                plt.ylabel("Pruning Rate")
                filename = f"{fairness_type}_stability_{pruning_strategy}.pdf"
                plt.tight_layout()
                plt.savefig(os.path.join(self.figure_save_dir, filename))
                plt.close()

    def plot_fairness_dual_axis(self, fairness_results):
        pruning_strategies = {
            f'{self.pruning_type}_clp': {'label': 'CLP', 'linestyle': '-'},
            f'{self.pruning_type}_dlp': {'label': 'DLP', 'linestyle': '--'}
        }

        pruning_rates = sorted(next(iter(fairness_results.values()))['max_min_gap'].keys())
        x_labels = [f"{r}" for r in pruning_rates]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        for strategy, style in pruning_strategies.items():
            max_min_values = [fairness_results[strategy]['quant_diff'][rate][self.num_models] for rate in pruning_rates]
            std_dev_values = [fairness_results[strategy]['cv'][rate][self.num_models] for rate in pruning_rates]

            ax1.plot(x_labels, max_min_values, label=f"Quant Diff ({style['label']})", linestyle=style['linestyle'],
                     color='tab:blue')
            ax2.plot(x_labels, std_dev_values, label=f"CV ({style['label']})", linestyle=style['linestyle'],
                     color='tab:orange')

        ax1.set_xlabel("Pruning Rate")
        ax1.set_ylabel("Quant Diff", color='tab:blue')
        ax2.set_ylabel("CV", color='tab:orange')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

        plt.title(f"Fairness (Ensemble Size = {self.num_models})")
        fig.tight_layout()
        filename = f"fairness_dual_axis_ensemble{self.num_models}.pdf"
        plt.savefig(os.path.join(self.figure_save_dir, filename))
        plt.close()

    def plot_fairness_change_heatmap(self, fairness_results):
        for pruning_strategy in [f'{self.pruning_type}_clp', f'{self.pruning_type}_dlp']:
            for fairness_type in ['max_min_gap', 'std_dev', 'cv', 'quant_diff', 'hard_easy', 'avg_change']:
                pruning_rates = sorted(fairness_results[pruning_strategy][fairness_type].keys())
                ensemble_sizes = list(range(1, self.num_models))  # from size 1 to N-1 (we compare i vs i+1)
                heatmap_data = []

                for pruning_rate in pruning_rates:
                    row = []
                    for i in ensemble_sizes:
                        fairness_i = fairness_results[pruning_strategy][fairness_type][pruning_rate][i]
                        fairness_ip1 = fairness_results[pruning_strategy][fairness_type][pruning_rate][i + 1]
                        delta = abs(fairness_ip1 - fairness_i)
                        row.append(delta)
                    heatmap_data.append(row)

                heatmap_data = np.array(heatmap_data)

                plt.figure(figsize=(10, 6))
                sns.heatmap(
                    heatmap_data, annot=True, fmt=".3f", center=0.0,
                    cmap="coolwarm", xticklabels=ensemble_sizes,
                    yticklabels=[f"{r}%" for r in pruning_rates]
                )
                plt.title(f"Δ {fairness_type.replace('_', ' ').title()} (i+1 - i) ({pruning_strategy})")
                plt.xlabel("Ensemble Size i (Δ = i+1 - i)")
                plt.ylabel("Pruning Rate")
                filename = f"{fairness_type}_change_heatmap_{pruning_strategy}.pdf"
                plt.tight_layout()
                plt.savefig(os.path.join(self.figure_save_dir, filename))
                plt.close()

    def compute_recall_instability(self, results):
        instability_results = {}
        for pruning_strategy in [f'{self.pruning_type}_clp', f'{self.pruning_type}_dlp']:
            instability_results[pruning_strategy] = {}

            for pruning_rate in sorted(results[pruning_strategy]['Recall'].keys()):
                instability_results[pruning_strategy][pruning_rate] = {}

                for i in range(1, self.num_models):  # comparing i vs i+1
                    recalls_i, recalls_ip1 = [], []

                    for class_id in range(self.num_classes):
                        mean_recall_i = np.mean([
                            results[pruning_strategy]['Recall'][pruning_rate][class_id][model_idx]
                            for model_idx in range(i)
                        ])
                        mean_recall_ip1 = np.mean([
                            results[pruning_strategy]['Recall'][pruning_rate][class_id][model_idx]
                            for model_idx in range(i + 1)
                        ])
                        recalls_i.append(mean_recall_i)
                        recalls_ip1.append(mean_recall_ip1)

                    # Sum of absolute differences
                    instability = np.sum(np.abs(np.array(recalls_ip1) - np.array(recalls_i)))
                    instability_results[pruning_strategy][pruning_rate][i] = instability / self.num_classes

        return instability_results

    def plot_recall_instability(self, instability_results):
        for pruning_strategy in [f'{self.pruning_type}_clp', f'{self.pruning_type}_dlp']:
            pruning_rates = sorted(instability_results[pruning_strategy].keys())
            ensemble_sizes = list(range(1, self.num_models))

            heatmap_data = np.array([
                [instability_results[pruning_strategy][rate][i] for i in ensemble_sizes]
                for rate in pruning_rates
            ])

            plt.figure(figsize=(10, 6))
            vmax = 0.019 if self.dataset_name == 'CIFAR100' else 0.005
            sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=vmax,
                        xticklabels=ensemble_sizes, yticklabels=[f"{r}%" for r in pruning_rates])
            plt.title(f"Recall Instability (|Δ Recall_c| summed) ({pruning_strategy})")
            plt.xlabel("Ensemble Size i (vs i+1)")
            plt.ylabel("Pruning Rate")
            plt.tight_layout()
            plt.savefig(os.path.join(self.figure_save_dir, f"recall_instability_{pruning_strategy}.pdf"))
            plt.close()

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

            fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharey=True)
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
            # Compute average recall over all pruning rates and models per class
            class_avg_recall = []
            for class_id in range(self.num_classes):
                recalls = [results[f'{self.pruning_type}_clp'][metric_name][0][class_id][model_idx]
                           for model_idx in range(self.num_models)]
                class_avg_recall.append(np.mean(recalls))

            # Identify 5 easiest and 5 hardest classes
            sorted_classes = np.argsort(class_avg_recall)
            selected_classes = list(sorted_classes[-5:][::-1]) + list(sorted_classes[:5])
            plot_indices = list(range(5)) + list(range(9, 4, -1))  # map to top and bottom row

            num_plots, ncols, nrows = 10, 5, 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharey='all')
            fig.suptitle(f"Class-Level {metric_name} Across Pruning Rates", fontsize=16)
            axes = axes.flatten()

            for idx, class_id in enumerate(selected_classes):
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

                pruning_clp = [pruned_percentages[f'{self.pruning_type}_clp'][p][class_id] for p in pruning_rates]
                pruning_dlp = [pruned_percentages[f'{self.pruning_type}_dlp'][p][class_id] for p in pruning_rates]

                ax = axes[plot_indices[idx]]
                ax.plot(pruning_clp, avg_metric_values_clp, marker='o', linestyle='-', color='blue', label='clp')
                ax.fill_between(pruning_clp, avg_metric_values_clp - std_metric_values_clp,
                                avg_metric_values_clp + std_metric_values_clp, color='blue', alpha=0.2)
                ax.set_xlabel("Pruning % (clp)", color='blue')
                ax.set_xlim(0, 100)
                ax.tick_params(axis='x', labelcolor='blue')

                ax2 = ax.twiny()
                ax2.plot(pruning_dlp, avg_metric_values_dlp, marker='s', linestyle='--', color='red', label='dlp')
                ax.fill_between(pruning_dlp, avg_metric_values_dlp - std_metric_values_dlp,
                                avg_metric_values_dlp + std_metric_values_dlp, color='red', alpha=0.2)
                ax2.set_xlabel("Pruning % (dlp)", color='red')
                ax2.set_xlim(0, 100)
                ax2.tick_params(axis='x', labelcolor='red')

                ax.set_ylabel(f"{metric_name}")
                ax.set_title(f"Class {class_id}")
                ax.grid(True, linestyle='--', alpha=0.6)

            # Hide any unused subplots
            for j in range(len(selected_classes), len(axes)):
                axes[j].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(self.figure_save_dir, f'Class_Level_{metric_name}.pdf'))
            plt.close()

    def compare_clp_with_dlp(self, results: Dict[str, Dict[str, Dict[int, Dict[int, Dict[int, int]]]]]):
        pruning_rates = sorted(results[f'{self.pruning_type}_clp']['Average Accuracy'].keys())
        x_labels = [f"{r}" for r in pruning_rates]
        print(f'x_labels - {x_labels}')

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
            plt.plot(x_labels, avg_metric_clp, marker='o', linestyle='-', color='blue', label='CLP')
            plt.fill_between(x_labels, avg_metric_clp - std_metric_clp, avg_metric_clp + std_metric_clp,
                             color='blue', alpha=0.2)
            # Plot DLP (Red Line)
            plt.plot(x_labels, avg_metric_dlp, marker='s', linestyle='--', color='red', label='DLP')
            plt.fill_between(x_labels, avg_metric_dlp - std_metric_dlp, avg_metric_dlp + std_metric_dlp,
                             color='red', alpha=0.2)  # Shaded region for std

            plt.xlabel("Percentage of samples removed from the dataset", fontsize=12)
            plt.ylabel(f"Average {metric_name}", fontsize=12)
            plt.legend()

            plt.savefig(os.path.join(self.figure_save_dir, f'{metric_name}_clp_vs_dlp.pdf'))

    def main(self):
        samples_per_class = load_results(os.path.join(self.results_dir, f'unclean{self.dataset_name}', f'alpha_1',
                                                      'samples_per_class.pkl'))
        models = self.load_models()
        print(f'Continuing with {self.num_models}.')
        pruned_percentages = self.compute_pruned_percentages(models)
        if self.num_classes == 10:
            self.plot_pruned_percentages(pruned_percentages[f'{self.pruning_type}_dlp'])

        # We only use test_loader so values of remove_noise, shuffle, and apply_augmentation doesn't matter below.
        _, _, test_loader, _ = load_dataset(self.dataset_name, False, False, True)

        # Evaluate ensemble performance
        if os.path.exists(os.path.join(self.results_dir, self.dataset_name, "ensemble_results.pkl")):
            print('Loading pre-computed ensemble results.')
            with open(os.path.join(self.results_dir, self.dataset_name, "ensemble_results.pkl"), 'rb') as f:
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
                        MCC = 0.0 if MCC_denominator == 0 else MCC_numerator / MCC_denominator

                        results[pruning_strategy]['F1'][pruning_rate][class_id][model_idx] = F1
                        results[pruning_strategy]['MCC'][pruning_rate][class_id][model_idx] = MCC
                        results[pruning_strategy]['Average Accuracy'][pruning_rate][class_id][model_idx] = accuracy
                        results[pruning_strategy]['Precision'][pruning_rate][class_id][model_idx] = precision
                        results[pruning_strategy]['Recall'][pruning_rate][class_id][model_idx] = recall

        self.plot_classwise_recall_vs_pruning(results)
        fairness_results = self.compute_fairness_metrics(results, samples_per_class)
        self.generate_fairness_table(fairness_results, os.path.join(self.results_dir, self.dataset_name))
        self.plot_fairness_stability(fairness_results)
        self.plot_fairness_dual_axis(fairness_results)
        self.plot_fairness_change_heatmap(fairness_results)
        instability_results = self.compute_recall_instability(results)
        self.plot_recall_instability(instability_results)
        self.compute_and_plot_recall_correlations_across_ensemble_sizes(results)
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

