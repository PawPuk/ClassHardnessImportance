"""This is the second visualization module that focuses on the results of experiment2.py

Main purpose:
*

Important information:
* Ensure that `num_models_per_dataset` and `num-datasets` from config.py have the same values as they did during
running experiment2.py!

"""
import argparse
import os
import pickle
from typing import Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from fiona.crs import defaultdict
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

from data import load_dataset
from config import get_config, ROOT
from utils import (compute_fairness_metrics, defaultdict_to_dict, generate_fairness_table, load_results,
                   obtain_results, plot_fairness_dual_axis, plot_fairness_stability)


class PerformanceVisualizer:
    """Encapsulates all the necessary methods to perform the visualization pertaining to the results of
    experiment2.py"""
    def __init__(self, dataset_name: str, alpha: str):
        """Initialize the Visualizer class responsible for visualizing the improvement in fairness from using different
        hardness-based resampling techniques on the controlled pruning case study (experiment2.py).

        :param dataset_name: Name of the dataset
        :param alpha: The module will produce visualization for this particular value of alpha (assuming that the
        experiments were run on this setting).
        """
        self.dataset_name = dataset_name
        self.alpha = int(alpha)

        config = get_config(args.dataset_name)
        self.num_classes = config['num_classes']
        self.num_epochs = config['num_epochs']
        self.num_training_samples = config['num_training_samples']
        self.num_test_samples = config['num_test_samples']
        self.num_models_per_dataset = config['num_models_per_dataset']
        self.num_datasets = config['num_datasets']

        self.figure_save_dir = os.path.join(ROOT, 'Figures/', args.dataset_name)
        self.results_dir = os.path.join(ROOT, "Results/")
        os.makedirs(self.figure_save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def load_models(self) -> Dict[str, Dict[int, Dict[int, List[str]]]]:
        """Used to load pretrained models."""
        models_dir = os.path.join(ROOT, "Models/")
        models_by_strategy, pruning_rate = defaultdict(lambda: defaultdict(lambda: defaultdict(list))), None

        for pruning_strategy in ['clp', 'random_dlp', 'holdout_dlp', 'SMOTE_dlp']:
            # Walk through each folder in the Models directory
            for root, dirs, files in os.walk(models_dir):
                # Ensure the dataset name matches exactly (avoid partial matches like "cifar10" in "cifar100")
                if f"{pruning_strategy}" in root and os.path.basename(root) == f'unclean{self.dataset_name}':
                    pruning_rate = int(root.split(pruning_strategy)[1].split("_alpha_")[0])
                    alpha = int(root.split("_alpha_")[1].split("/")[0])

                    for file in files:
                        if file.endswith(".pth") and f"_epoch_{self.num_epochs}" in file and alpha == self.alpha:
                            dataset_index = int(file.split("_")[1])
                            model_index = int(file.split("_")[3])
                            if dataset_index >= self.num_datasets or model_index >= self.num_models_per_dataset:
                                raise Exception('The `num_datasets` and `num_models_per_dataset` in config.py needs to '
                                                'have the same values as it when running experiment3.py.')

                            model_path = os.path.join(root, file)
                            models_by_strategy[pruning_strategy][pruning_rate][dataset_index].append(model_path)
                    if len(models_by_strategy[pruning_strategy][pruning_rate]) > 0:
                        for i in range(1, len(models_by_strategy[pruning_strategy][pruning_rate])):  # Sanity check
                            assert len(models_by_strategy[pruning_strategy][pruning_rate][0]) == \
                                   len(models_by_strategy[pruning_strategy][pruning_rate][i])
                        print(
                            f"Loaded {len(models_by_strategy[pruning_strategy][pruning_rate])} ensembles of models for "
                            f"strategies {pruning_strategy} and pruning rate {pruning_rate}, with each ensemble having "
                            f"{len(models_by_strategy[pruning_strategy][pruning_rate][0])} models.")

            # Load models trained on the full dataset (no pruning)
            full_dataset_dir = os.path.join(models_dir, "none", f'unclean{self.dataset_name}')
            if os.path.exists(full_dataset_dir):
                for file in os.listdir(full_dataset_dir):
                    if file.endswith(".pth") and f"_epoch_{self.num_epochs}" in file:
                        model_path = os.path.join(full_dataset_dir, file)
                        models_by_strategy[pruning_strategy][0][0].append(model_path)

            print(f"Models loaded by pruning rate for {pruning_strategy} on {self.dataset_name}")

        print(models_by_strategy.keys())
        print(models_by_strategy)
        return defaultdict_to_dict(models_by_strategy)

    def compute_pruned_percentages(self, models_by_rate: Dict[str, Dict[int, Dict[int, List[str]]]]
                                   ) -> Dict[str, Dict[int, List[float]]]:
        """Computes the percentage of samples that were pruned per class for each of the pruning rate and pruning
        strategy."""
        pruned_percentages = {}
        for pruning_strategy in models_by_rate.keys():
            pruning_rates = models_by_rate[pruning_strategy].keys()
            pruned_percentages[pruning_strategy] = {pruning_rate: [] for pruning_rate in pruning_rates}

            # Iterate over each pruning rate in models_by_rate
            for pruning_rate in pruning_rates:
                if int(pruning_rate) != 0:
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
        print('Pruned percentages: ', pruned_percentages)

        return pruned_percentages

    def plot_pruned_percentages(self, pruned_percentages: Dict[int, List[float]]):
        """Visualizes the differences between pruned samples for class- and dataset-level pruning for the top-5 easiest
        and hardest classes."""
        plt.figure(figsize=(10, 6))

        pruning_rates = sorted(pruned_percentages.keys())
        class_means = {}
        for class_idx in range(self.num_classes):
            class_pruning_percentages = [pruned_percentages[pr][class_idx] for pr in pruning_rates]
            class_means[class_idx] = np.mean(class_pruning_percentages)

        sorted_classes = sorted(class_means.items(), key=lambda x: x[1])
        easiest = [cls_idx for cls_idx, _ in sorted_classes[-5:]]
        hardest = [cls_idx for cls_idx, _ in sorted_classes[:5]]

        selected_classes = easiest + hardest
        for class_idx in selected_classes:
            class_pruning_percentages = [pruned_percentages[pruning_rate][class_idx] for pruning_rate in pruning_rates]
            plt.plot(pruning_rates[1:], class_pruning_percentages[1:], marker='o')

        plt.xlabel("Percentage of samples removed from the dataset (DLP rate)")
        plt.ylabel("Class-Level Pruning Percentage (%)")
        plt.grid(True)
        plt.savefig(os.path.join(self.figure_save_dir, f'class_vs_dataset_pruning_values.pdf'))
        # TODO: maybe modify later to compute percentages here to include more points for smoother lines...

    def plot_classwise_recall_vs_pruning(
            self,
            results: Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]]
    ):
        """This visualization goes through all pruning rates and for which visually compares the per-class recall,
        averaged across datasets and models, across different pruning strategies. The idea is to introduce another way
        to compare the effect of different pruning strategies on performance on easy and hard classes - lower/higher
        performance on easy/hard classes for clp-based results than dlp-based ones."""
        def flatten(d):
            """Helper function for transforming Dict[int, Dict[int, float]] into List[float]"""
            return [v for inner in d.values() for v in inner.values()]

        pruning_rates = list(results['Recall']['clp'].keys())
        print('Pruning rates:', pruning_rates)
        zero_rate = 0.0 if 0.0 in pruning_rates else 0
        pruning_rate_no_zero = sorted([pr for pr in pruning_rates if pr != zero_rate])
        colors = matplotlib.colormaps["tab10"]

        fig, axes = plt.subplots(ncols=len(pruning_rate_no_zero), figsize=(4 * len(pruning_rate_no_zero), 8),
                                 sharey='all')
        for ax, pruning_rate in zip(axes, pruning_rate_no_zero):
            clp_recalls = [np.mean(flatten(results['Recall']['clp'][pruning_rate][cls]))
                           for cls in range(self.num_classes)]
            random_dlp_recalls = [np.mean(flatten(results['Recall']['random_dlp'][pruning_rate][cls]))
                                  for cls in range(self.num_classes)]
            SMOTE_dlp_recalls = [np.mean(flatten(results['Recall']['SMOTE_dlp'][pruning_rate][cls]))
                                 for cls in range(self.num_classes)]
            holdout_dlp_recalls = [np.mean(flatten(results['Recall']['holdout_dlp'][pruning_rate][cls]))
                                   for cls in range(self.num_classes)]

            sorted_classes = np.argsort(clp_recalls)[::-1]
            clp_sorted = np.array(clp_recalls)[sorted_classes]
            random_dlp_sorted = np.array(random_dlp_recalls)[sorted_classes]
            SMOTE_dlp_sorted = np.array(SMOTE_dlp_recalls)[sorted_classes]  # noqa
            holdout_dlp_sorted = np.array(holdout_dlp_recalls)[sorted_classes]

            ax.plot(clp_sorted, label=f'CLP', color=colors(0))
            ax.plot(random_dlp_sorted, label=f'random DLP', color=colors(1))
            ax.plot(SMOTE_dlp_sorted, label=f'SMOTE DLP', color=colors(2))
            ax.plot(holdout_dlp_sorted, label=f'holdout DLP', color=colors(3))

            ax.set_title(f'Class-wise Recall @ pruning_rate={pruning_rate}')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        fig.supxlabel('Class (sorted by baseline recall)')
        fig.supylabel('Recall')
        plt.tight_layout()
        filename = f"class_level_recall_changes_investigation.pdf"
        plt.savefig(os.path.join(self.figure_save_dir, filename))

    def compute_and_plot_recall_correlations_across_ensemble_sizes(
            self,
            results: Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]]
    ):
        """This visualization measures the Spearman Rank and Pearson correlations between the class-level recall values
         averaged over i x i ensemble and (i+1) x (i+1) ensemble - (num_datasets) x (num_models_per_dataset). The idea
         is to see if we have to use more models datasets to ensure statistical significance of class-level recalls."""
        def compute_mean_recall(class_id, size):
            """Helper function"""
            if pruning_rate == 0:
                return np.mean([results['Recall'][pruning_strategy][pruning_rate][class_id][0][m]
                                for m in range(size * size)])
            else:
                return np.mean([results['Recall'][pruning_strategy][pruning_rate][class_id][d][m]
                                for d in range(size) for m in range(size)])

        for pruning_strategy in results['Recall'].keys():
            sorted_pruning_rates = sorted(results['Recall'][pruning_strategy].keys())
            pearson_matrix, spearman_matrix, pearson_pval_matrix, spearman_pval_matrix = [], [], [], []
            for pruning_rate in sorted_pruning_rates:
                pearson_row, spearman_row, pearson_pval_row, spearman_pval_row = [], [], [], []
                for i in range(1, min(self.num_datasets, self.num_models_per_dataset)):
                    # Recalls for ensemble of size i x i
                    recall_i = [compute_mean_recall(c, i) for c in range(self.num_classes)]
                    # Recalls for ensemble of size (i + 1) x (i + 1)
                    recall_ip1 = [compute_mean_recall(c, i + 1) for c in range(self.num_classes)]

                    pearson_corr, pearson_pval = pearsonr(recall_i, recall_ip1)
                    spearman_corr, spearman_pval = spearmanr(recall_i, recall_ip1)  # noqa
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

            fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharey='all')
            x_labels = list(range(1, min(self.num_datasets, self.num_models_per_dataset)))
            y_labels = [f"{r}%" for r in sorted_pruning_rates]
            heatmaps = [
                (pearson_matrix, "Pearson Correlation", axes[0, 0]),
                (spearman_matrix, "Spearman Correlation", axes[0, 1]),
                (pearson_pval_matrix, "Pearson p-value", axes[1, 0]),
                (spearman_pval_matrix, "Spearman p-value", axes[1, 1]),
            ]
            for mat, title, ax in heatmaps:
                sns.heatmap(
                    mat, xticklabels=x_labels, yticklabels=y_labels, annot=True, fmt=".2f",
                    cmap="YlGnBu" if "Correlation" in title else "OrRd", ax=ax, vmin=0.0, vmax=1.0
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

    def plot_class_level_results(
            self,
            results: Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]],
            pruned_percentages: Dict[str, Dict[int, List[float]]]
    ):
        """This visualization also compares the class-level recalls across different pruning strategies and pruning
        thresholds, but in a different manner. This time we focus on 10 most extreme classes (5 easiest and 5 hardest)
        and also show the differences in pruned thresholds (e.g., pruning threshold of 20% leads to removal of way more
         samples on the hardest classes)."""
        def compute_statistic(pruning_strategy, reducer):
            """Helper function"""
            values = []
            for p in pruning_rates:
                if p == 0:
                    stat = reducer([
                        results[metric_name][pruning_strategy][p][class_id][0][model_idx]
                        for model_idx in range(self.num_models_per_dataset * self.num_models_per_dataset)
                        if results[metric_name][pruning_strategy][p][class_id][0][model_idx] != 0.0
                    ])
                else:
                    stat = reducer([
                        results[metric_name][pruning_strategy][p][class_id][dataset_idx][model_idx]
                        for dataset_idx in range(self.num_datasets)
                        for model_idx in range(self.num_models_per_dataset)
                        if results[metric_name][pruning_strategy][p][class_id][dataset_idx][model_idx] != 0.0
                    ])
                values.append(stat)
            return np.array(values)

        pruning_rates = sorted(results['Recall']['clp'].keys())
        colors = matplotlib.colormaps["tab10"]
        for metric_name in results.keys():
            # Identify 5 easiest and 5 hardest classes (based on Recall values of a single model)
            hardness = [results['Recall']['clp'][0][cls][0][0] for cls in range(self.num_classes)]
            sorted_classes = np.argsort(np.array(hardness))
            selected_classes = list(sorted_classes[-5:][::-1]) + list(sorted_classes[:5])
            plot_indices = list(range(5)) + list(range(9, 4, -1))  # map to top and bottom row

            num_plots, num_cols, num_rows = 10, 5, 2
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), sharey='all')
            fig.suptitle(f"Class-Level {metric_name} Across Pruning Rates", fontsize=16)
            axes = axes.flatten()

            for idx, class_id in enumerate(selected_classes):
                avg_metric_values_clp = compute_statistic('clp', np.mean)
                # std_metric_values_clp = compute_statistic('clp', np.std)
                pruning_clp = [pruned_percentages['clp'][p][class_id] for p in pruning_rates]

                ax = axes[plot_indices[idx]]
                ax.plot(pruning_clp, avg_metric_values_clp, marker='o', linestyle='-', color=colors(0), label='clp')
                """ax.fill_between(pruning_clp, avg_metric_values_clp - std_metric_values_clp,
                                avg_metric_values_clp + std_metric_values_clp, color=colors(0), alpha=0.2)"""
                ax.set_xlabel("Pruning %")
                # ax.set_xlim(0, 100)

                for i, strategy in enumerate(['random_dlp', 'SMOTE_dlp', 'holdout_dlp']):
                    avg_metric_values_dlp = compute_statistic(strategy, np.mean)
                    # std_metric_values_dlp = compute_statistic(strategy, np.std)
                    pruning_dlp = [pruned_percentages[strategy][p][class_id] for p in pruning_rates]

                    ax.plot(pruning_dlp, avg_metric_values_dlp, marker='o', linestyle='-', color=colors(i + 1),
                            label=strategy)
                    """ax.fill_between(pruning_dlp, avg_metric_values_dlp - std_metric_values_dlp,
                                    avg_metric_values_dlp + std_metric_values_dlp, color=colors(i + 1), alpha=0.2)"""

                ax.set_ylabel(f"{metric_name}")
                ax.set_title(f"Class {class_id}")
                ax.grid(True, linestyle='--', alpha=0.6)
                lines_1, labels_1 = ax.get_legend_handles_labels()
                ax.legend(lines_1, labels_1, ncol=2)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(self.figure_save_dir, f'Class_Level_{metric_name}.pdf'))
            plt.close()

    def compare_clp_with_dlp(
            self,
            results: Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]]
    ):
        """This visualization provides the final comparison across pruning techniques. This time the comparison is
        performed at dataset leve. On one hand it provides the least information, but at the same time this should be
        the easiest to understand due to simplicity and clarity."""
        def compute_statistic(pruning_strategy, reducer):
            """Helper function"""
            values = []
            for p in pruning_rates:
                if p == 0:
                    # Special structure: results[metric][strategy][0][class_id][0][model_idx]
                    stat = reducer([
                        results[metric_name][pruning_strategy][p][class_id][0][model_idx]
                        for class_id in range(self.num_classes)
                        for model_idx in range(self.num_models_per_dataset * self.num_models_per_dataset)
                        if results[metric_name][pruning_strategy][p][class_id][0][model_idx] != 0.0
                    ])
                else:
                    # Normal structure: results[metric][strategy][p][class_id][dataset_idx][model_idx]
                    stat = reducer([
                        results[metric_name][pruning_strategy][p][class_id][dataset_idx][model_idx]
                        for class_id in range(self.num_classes)
                        for dataset_idx in range(self.num_datasets)
                        for model_idx in range(self.num_models_per_dataset)
                        if results[metric_name][pruning_strategy][p][class_id][dataset_idx][model_idx] != 0.0
                    ])
                values.append(stat)
            return np.array(values)

        pruning_rates = sorted(results['Recall']['clp'].keys())
        x_labels = [f"{r}" for r in pruning_rates]
        colors = matplotlib.colormaps["tab10"]

        for metric_name in results.keys():
            avg_metric_clp = compute_statistic('clp', np.mean)
            std_metric_clp = compute_statistic('clp', np.std)

            plt.figure(figsize=(8, 6))
            plt.plot(x_labels, avg_metric_clp, marker='o', linestyle='-', color=colors(0), label='CLP')
            plt.fill_between(x_labels, avg_metric_clp - std_metric_clp, avg_metric_clp + std_metric_clp,
                             color=colors(0), alpha=0.2)

            for i, strategy in enumerate(['holdout_dlp']):
                avg_metric_dlp = compute_statistic(strategy, np.mean)
                std_metric_dlp = compute_statistic(strategy, np.std)
                if metric_name == 'Recall':
                    print(avg_metric_dlp)

                plt.plot(x_labels, avg_metric_dlp, marker='s', linestyle='--', color=colors(i + 1),
                         label='Holdout' if 'dlp' in strategy else 'CLP')
                plt.fill_between(x_labels, avg_metric_dlp - std_metric_dlp, avg_metric_dlp + std_metric_dlp,
                                 color=colors(i + 1), alpha=0.2)

            plt.xlabel("Percentage of samples removed from the dataset", fontsize=12)
            plt.ylabel(f"Average {metric_name}", fontsize=12)
            plt.legend(ncol=2)
            plt.savefig(os.path.join(self.figure_save_dir, f'{metric_name}_clp_vs_dlp.pdf'))

    def compute_recall_instability(
            self,
            results: Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Computes instability of class-level recall values when increasing ensemble size
        (max dataset_idx, max model_idx) from i to i+1.

        Returns:
            Dict[str, Dict[int, Dict[str, float]]]
            Format: results[pruning_strategy][i] = {"mean": float, "std": float}
        """
        def compute_statistic(j, class_id, reducer):
            """Helper function"""
            return np.array([
                reducer([
                    results[pruning_strategy][75][class_id][dataset_idx][model_idx]
                    for dataset_idx in range(j)
                    for model_idx in range(j)
                    if results[pruning_strategy][75][class_id][dataset_idx][model_idx] != 0.0
                ])
            ])

        instability_results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for pruning_strategy in ['clp', 'random_dlp', 'holdout_dlp', 'SMOTE_dlp']:
            for i in range(1, min(self.num_datasets, self.num_models_per_dataset)):
                recalls_i = np.array([compute_statistic(i, class_id, np.mean) for class_id in range(self.num_classes)])
                recalls_ip1 = np.array(
                    [compute_statistic(i + 1, class_id, np.mean) for class_id in range(self.num_classes)])

                # per-class instability
                instability_per_class = np.abs(recalls_ip1 - recalls_i).flatten()

                instability_results[pruning_strategy][i]["mean"] = float(np.mean(instability_per_class))
                instability_results[pruning_strategy][i]["std"] = float(np.std(instability_per_class))

        return defaultdict_to_dict(instability_results)

    def plot_recall_instability(self, instability_results: Dict[str, Dict[int, Dict[str, float]]]):
        """Line plot: Recall instability (mean ± std) vs ensemble size. One curve per pruning strategy."""
        plt.figure(figsize=(10, 6))
        colors = matplotlib.colormaps["tab10"]

        for idx, pruning_strategy in enumerate(['clp', 'random_dlp', 'holdout_dlp', 'SMOTE_dlp']):
            ensemble_sizes = sorted(instability_results[pruning_strategy].keys())
            means = [instability_results[pruning_strategy][i]["mean"] for i in ensemble_sizes]
            stds = [instability_results[pruning_strategy][i]["std"] for i in ensemble_sizes]

            plt.plot(ensemble_sizes, means, label=pruning_strategy, color=colors(idx), marker="o")
            plt.fill_between(ensemble_sizes,
                             np.array(means) - np.array(stds),
                             np.array(means) + np.array(stds),
                             color=colors(idx), alpha=0.2)

        plt.xlabel("Ensemble Size (i vs i+1)", fontsize=12)
        plt.xticks([1, 2, 3, 4])
        plt.ylabel("Recall Instability (mean ± std)", fontsize=12)
        plt.title("Recall Instability across Ensemble Sizes", fontsize=14)
        plt.legend(loc="upper right", ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_save_dir, "recall_instability_all.pdf"))
        plt.close()

    def main(self):
        """Main method for producing the visualizations."""
        models = self.load_models()
        pruned_percentages = self.compute_pruned_percentages(models)
        self.plot_pruned_percentages(pruned_percentages[f'random_dlp'])
        _, _, test_loader, _ = load_dataset(self.dataset_name, apply_augmentation=True)

        results = obtain_results(os.path.join(self.results_dir, self.dataset_name), self.num_classes, test_loader,
                                 "ensemble_results.pkl", models)
        self.plot_classwise_recall_vs_pruning(results)
        self.compute_and_plot_recall_correlations_across_ensemble_sizes(results)
        self.plot_class_level_results(results, pruned_percentages)
        self.compare_clp_with_dlp(results)

        instability_results = self.compute_recall_instability(results['Recall'])
        self.plot_recall_instability(instability_results)

        samples_per_class = load_results(os.path.join(self.results_dir, f'unclean{self.dataset_name}', f'alpha_1',
                                                      'samples_per_class.pkl'))
        pruning_strategies = ['clp', 'random_dlp', 'holdout_dlp', 'SMOTE_dlp']
        fairness_results = compute_fairness_metrics(results, samples_per_class, pruning_strategies, self.num_classes,
                                                    min(self.num_datasets, self.num_models_per_dataset))
        generate_fairness_table(fairness_results, os.path.join(self.results_dir, self.dataset_name),
                                min(self.num_datasets, self.num_models_per_dataset), 'pruning')
        plot_fairness_stability(fairness_results, self.figure_save_dir)
        plot_fairness_dual_axis(fairness_results, self.figure_save_dir, 'pruning')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., 'CIFAR10')")
    parser.add_argument('--alpha', type=int, default=1, help='Value of the alpha parameter for which we want to produce'
                                                             ' visualizations (see experiment2.py for details).')
    args = parser.parse_args()
    PerformanceVisualizer(args.dataset_name, args.alpha).main()
