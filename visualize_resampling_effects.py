"""This is the third visualization module that focuses on the results of experiment3.py

Main purpose:
* Compare Recall, Precision, F1 Score and MCC between ensembles trained on original datasets, and the ones trained on
data obtained through hardness-based resampling.
* Measure and visualize the changes to fairness coming from hardness-based resampling using metrics such as
Coefficient of Variation, Standard Deviation of metric values and others.

Important information:
* Ensure that `num_models_per_dataset` and `num_datasets` from config.py have the same values as they did during
running experiment3.py!
"""
import argparse
from collections import defaultdict
import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from config import get_config, ROOT
from data import load_dataset
from utils import (compute_fairness_metrics, defaultdict_to_dict, generate_t_test_table_for_resampling, load_results,
                   obtain_results, perform_paired_t_tests, plot_fairness_dual_axis, plot_fairness_stability)


class ResamplingVisualizer:
    """Encapsulates all the necessary methods to perform the visualization pertaining to the results of
    experiments3.py."""
    def __init__(self, dataset_name, remove_noise):
        """Initialize the Visualizer class responsible for visualizing the improvement in fairness from using different
         hardness-based resampling techniques on the full data (experiment3.py).

         :param dataset_name: Name of the dataset
         :param remove_noise: If experiment3.py was run with this parameter raised than it also has to be raised here.
        Otherwise, keep it as it.
        """
        self.dataset_name = dataset_name
        self.clean_data = 'clean' if remove_noise else 'unclean'

        config = get_config(args.dataset_name)
        self.num_classes = config['num_classes']
        self.num_training_samples = config['num_training_samples']
        self.num_test_samples = config['num_test_samples']
        self.num_epochs = config['num_epochs']
        self.num_models_per_dataset = config['num_models_per_dataset']
        self.num_datasets = config['num_datasets']

        self.figure_save_dir = os.path.join(ROOT, 'Figures/', dataset_name)
        self.hardness_save_dir = os.path.join(ROOT, f"Results/")

        for save_dir in self.figure_save_dir, self.hardness_save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def load_models(self) -> Dict[Tuple[str, str], Dict[int, Dict[int, List[str]]]]:
        """Used to load pretrained models."""
        models_dir = os.path.join(ROOT, "Models")
        models_by_strategy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for root, dirs, files in os.walk(models_dir):
            if 'over_' in root and os.path.basename(root) == f'{self.clean_data}{self.dataset_name}':
                oversampling_strategy = root.split("over_")[1].split("_under_")[0]
                undersampling_strategy = root.split("_under_")[1].split("_alpha_")[0]
                alpha = root.split("_alpha_")[1].split("_hardness_")[0]
                key = (oversampling_strategy, undersampling_strategy)

                for file in files:
                    if file.endswith(".pth") and f"_epoch_{self.num_epochs}" in file:
                        # For cases where the number of trained models is above robust_ensemble_size
                        dataset_index = int(file.split("_")[1])
                        model_index = int(file.split("_")[3])
                        if dataset_index >= self.num_datasets or model_index >= self.num_models_per_dataset:
                            raise Exception('The `num_datasets` and `num_models_per_dataset` in config.py needs to '
                                            'have the same values as it when running experiment3.py.')

                        model_path = os.path.join(root, file)
                        models_by_strategy[key][alpha][dataset_index].append(model_path)
                if len(models_by_strategy[key][alpha]) > 0:
                    print(f"Loaded {len(models_by_strategy[key][alpha])} ensembles of models for strategies {key} and "
                          f"alpha {alpha}, with each ensemble having {len(models_by_strategy[key][alpha][0])} models.")
                    for i in range(1, len(models_by_strategy[key][alpha])):  # Sanity check
                        assert len(models_by_strategy[key][alpha][0]) == len(models_by_strategy[key][alpha][i])

        # Also load models trained on the full dataset (no resampling)
        full_dataset_dir = os.path.join(models_dir, "none", f"{self.clean_data}{self.dataset_name}")
        if os.path.exists(full_dataset_dir):
            key = ('none', 'none')
            for file in os.listdir(full_dataset_dir):
                if file.endswith(".pth") and f"_epoch_{self.num_epochs}" in file:
                    model_path = os.path.join(full_dataset_dir, file)
                    models_by_strategy[key][1][0].append(model_path)

        print(f"Loaded {len(models_by_strategy.keys())} ensembles for {self.dataset_name}.")
        return defaultdict_to_dict(models_by_strategy)

    def load_sample_allocations(self) -> Dict[int, List[int]]:
        """Extracts the sample allocations after resampling for different alphas"""
        sample_allocations = {}
        for root, dirs, files in os.walk(self.hardness_save_dir):
            if f"unclean{self.dataset_name}/" in root and 'alpha_' in root:
                alpha = int(root.split('alpha_')[-1])
                for file in files:
                    file_path = os.path.join(root, file)
                    sample_allocations[alpha] = load_results(file_path)
        print(f'Loaded info on class-wise sample allocation after hardness-based resampling:\n\t{sample_allocations}')
        return sample_allocations

    def visualize_resampling_results(self, sample_allocation: Dict[int, List[int]]) -> Tuple[int, List[int]]:
        """This visualization shows the effects of resampling. It clearly shows how many samples were removed from each
        class due to hardness-based resampling (the classes are sorted for clarity). It also computes the number of
        easy classes in the dataset - the classes that are undersampled during hardness-based resampling."""
        alpha_values = sorted(map(int, sample_allocation.keys()))
        avg_count = np.mean(sample_allocation[alpha_values[0]])
        min_count = min(min(samples) for samples in sample_allocation.values())
        max_count = max(max(samples) for samples in sample_allocation.values())

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhspan(min_count, avg_count, color='green', alpha=0.15)
        ax.axhline(y=avg_count, color='blue', linestyle='--', linewidth=2, label='α=0')
        ax.axhspan(avg_count, max_count, color='red', alpha=0.15)

        for alpha in sample_allocation.keys():
            norm = plt.Normalize(min(alpha_values), max(alpha_values))
            cmap = plt.get_cmap("Greys")
            grey_shade = cmap(0.5 + 0.5 * norm(alpha))  # Scale to [0.5, 1] for visual clarity

            sorted_indices = np.argsort(sample_allocation[alpha])
            sorted_counts = np.array(sample_allocation[alpha])[sorted_indices]
            ax.plot(range(len(sorted_counts)), sorted_counts, color=grey_shade, label=f'α={alpha}', linewidth=2)

        ax.set_xlabel('Classes sorted based on hardness (hardest to the right)')
        ax.set_xticklabels([])
        ax.set_xticks(np.arange(1, self.num_classes + 1))
        ax.set_ylabel('Class-wise sample count after resampling')
        ax.set_title(self.dataset_name)
        ax.legend()
        fig.savefig(os.path.join(self.figure_save_dir, 'sorted_resampled_dataset.pdf'))

        number_of_easy_classes = sum([sample_allocation[alpha_values[0]][cls] <= avg_count
                                      for cls in range(len(sample_allocation[alpha_values[0]]))])
        print(f'Identified {number_of_easy_classes} easy classes in this dataset.')
        return number_of_easy_classes, alpha_values

    def plot_all_metrics_sorted(
            self,
            results: Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]],
            class_order: NDArray,
            base_metric: str,
            number_of_easy_classes: int,
            strategy: str,
            alpha_values: List[int]
    ):
        """This visualization plots the metric (e.g., Recall, F1, ...) values sorted based on the hardness on the
        untouched dataset. We want to see whether hardness-based imbalance visibly improves the performance on hard
        classes."""
        def flatten(d):
            """Helper function for transforming Dict[int, Dict[int, float]] into List[float]"""
            return [v for inner in d.values() for v in inner.values()]

        base_strategy, base_alpha = ('none', 'none'), 1

        plt.figure(figsize=(12, 8))
        reordered_base_values = [np.mean(flatten(results[base_metric][base_strategy][base_alpha][class_id]))
                                 for class_id in class_order]
        plt.plot(range(len(class_order)), [v for v in reordered_base_values], color='black', linestyle='-',
                 linewidth=4, label="α=1")

        for alpha in results[base_metric][strategy].keys():
            norm = plt.Normalize(min(alpha_values), max(alpha_values))
            cmap = plt.get_cmap("Greys")
            grey_shade = cmap(0.5 + 0.5 * norm(alpha))
            reordered_values = [np.mean(flatten(results[base_metric][strategy][alpha][class_id]))
                                for class_id in class_order]
            plt.plot(range(len(class_order)), reordered_values, color=grey_shade, lw=2, linestyle='--',
                     label=f'"α={alpha}"')

        plt.axvspan(0, number_of_easy_classes - 0.5, color='green', alpha=0.15)
        plt.axvspan(number_of_easy_classes - 0.5, len(class_order), color='red', alpha=0.15)

        plt.xlabel("Classes sorted based on hardness (hardest to the right)", fontsize=12)
        plt.xticks(np.arange(0, len(class_order)), labels=[])
        plt.ylabel(f"Class-wise {base_metric} on balanced dataset", fontsize=12)
        plt.title(self.dataset_name, fontsize=14)
        plt.grid(True, alpha=0.6, axis='y')
        plt.legend()
        plt.savefig(os.path.join(self.figure_save_dir, f'{base_metric}_{strategy}_effects.pdf'),
                    bbox_inches='tight')

    def plot_metric_changes(
            self,
            results: Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]],
            class_order: NDArray,
            base_metric: str,
            number_of_easy_classes: int,
            strategy: str,
            alpha_values: List[int]
    ):
        """This visualization completes the plot_all_metrics_sorted() method by focusing on the differences between the
        performance on normal dataset and the resampled dataset. This is useful as the visualization from
        plot_all_metrics_sorted() can be unclear."""
        def flatten(d):
            """Helper function for transforming Dict[int, Dict[int, float]] into List[float]"""
            return [v for inner in d.values() for v in inner.values()]

        base_strategy, base_alpha, value_changes = (('none', 'none'), 1, defaultdict(lambda: defaultdict(list)))

        plt.figure(figsize=(12, 8))

        for alpha in results[base_metric][strategy].keys():
            value_changes[strategy][alpha] = [
                float(np.mean(flatten(results[base_metric][strategy][alpha][class_id]))) -
                float(np.mean(flatten(results[base_metric][base_strategy][base_alpha][class_id])))
                for class_id in class_order
            ]
            norm = plt.Normalize(min(alpha_values), max(alpha_values))
            cmap = plt.get_cmap("Greys")
            grey_shade = cmap(0.5 + 0.5 * norm(alpha))
            plt.axhline(color='black', linewidth=2)  # To accentuate X-axis
            plt.plot(range(self.num_classes), value_changes[strategy][alpha], color=grey_shade,
                     linewidth=2, label=f"α={alpha}")

            print(f'{alpha}, {strategy} - easy mean: '
                  f'{np.mean(value_changes[strategy][alpha][:number_of_easy_classes])}'
                  f', hard mean:  {np.mean(value_changes[strategy][alpha][number_of_easy_classes:])}, easy std: '
                  f'{np.std(value_changes[strategy][alpha][:number_of_easy_classes])}, hard std:'
                  f'{np.std(value_changes[strategy][alpha][number_of_easy_classes:])}')

        plt.axvspan(0, number_of_easy_classes - 0.5, color='green', alpha=0.15)
        plt.axvspan(number_of_easy_classes - 0.5, len(class_order), color='red', alpha=0.15)

        plt.xlabel("Classes sorted based on hardness (hardest to the right)", fontsize=12)
        plt.xticks(np.arange(0, len(class_order)), labels=[])
        plt.ylabel(f"Class-wise {base_metric} change after resampling", fontsize=12)
        plt.title(self.dataset_name, fontsize=12)
        plt.grid(True, alpha=0.6, axis='y')
        plt.legend()
        plt.savefig(os.path.join(self.figure_save_dir, f'{base_metric}_changes_due_to_{strategy}.pdf'),
                    bbox_inches='tight')

    def main(self):
        """Main method for producing the visualizations."""
        results_dir = os.path.join(ROOT, "Results", self.dataset_name)
        models = self.load_models()
        _, _, test_loader, _ = load_dataset(self.dataset_name, apply_augmentation=True)

        results = obtain_results(results_dir, self.num_classes, test_loader, "resampling_results.pkl", models)

        samples_per_class = self.load_sample_allocations()
        number_of_easy_classes, alpha_values = self.visualize_resampling_results(samples_per_class)
        # the alpha value (in samples_per_class[alpha] below) doesn't matter as long as its valid
        class_order = np.argsort(samples_per_class[1])
        for base_metric in ['F1', 'MCC', 'Precision', 'Recall']:
            for strategy in results['Recall'].keys():
                self.plot_all_metrics_sorted(results, class_order, base_metric, number_of_easy_classes, strategy,
                                             alpha_values)
                print(f'{"-" * 70}\n\tResults of {strategy} for {base_metric}:\n{"-" * 70}')
                self.plot_metric_changes(results, class_order, base_metric, number_of_easy_classes, strategy,
                                         alpha_values)

        for metric_name in ['Recall', 'F1', 'MCC', 'Precision']:
            for strategy in results[metric_name]:
                if strategy is not ('none', 'none'):
                    results[metric_name][strategy][0] = results[metric_name][('none', 'none')][1]
            del results[metric_name][('none', 'none')]
        resampling_strategies = list(results['Recall'].keys())

        fairness_results = compute_fairness_metrics(results, samples_per_class[1], resampling_strategies,
                                                    self.num_classes)

        plot_fairness_stability(fairness_results, self.figure_save_dir)
        plot_fairness_dual_axis(fairness_results, self.figure_save_dir, 'resampling')
        t_test_results = perform_paired_t_tests(fairness_results, 'resampling')
        generate_t_test_table_for_resampling(t_test_results, self.figure_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (e.g., 'CIFAR10')")
    parser.add_argument('--remove_noise', action='store_true', help='Raise this flag to remove noise from the data.')

    args = parser.parse_args()
    ResamplingVisualizer(args.dataset_name, args.remove_noise).main()

# TODO: Plot class-level recall vs AUM to show that the correlation exists but it's not 1 to 1.
# TODO: If time allows rerun the experiments with Recall being used as the hardness estimator for resampling.
