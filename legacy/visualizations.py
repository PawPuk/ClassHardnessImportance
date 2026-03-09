from collections import defaultdict
import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_all_metrics_sorted(
        results: Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]],
        class_order: NDArray,
        base_metric: str,
        number_of_easy_classes: int,
        strategy: str,
        alpha_values: List[int],
        dataset_name: str,
        figure_save_dir: str
):
    """Plot the Precision/Recall sorted by AUM-based hardness. This plots a line per alpha. However, the output is
    slightly clogged, so we left it out (."""
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
    plt.title(dataset_name, fontsize=14)
    plt.grid(True, alpha=0.6, axis='y')
    plt.legend()
    plt.savefig(os.path.join(figure_save_dir, f'{base_metric}_{strategy}_effects.pdf'),
                bbox_inches='tight')


def plot_metric_changes(
        results: Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]],
        class_order: NDArray,
        base_metric: str,
        number_of_easy_classes: int,
        strategy: str,
        alpha_values: List[int],
        num_classes: int,
        dataset_name: str,
        figure_save_dir: str
):
    """The plot produced by this function complements the one from the function above, as it shows only the class-level
    differences in Precision/Recall brought by applying hardness-based imbalance. However, this still isn't as clear as
    our delta metrics, so we didn't include this Figure."""
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
        plt.plot(range(num_classes), value_changes[strategy][alpha], color=grey_shade,
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
    plt.title(dataset_name, fontsize=12)
    plt.grid(True, alpha=0.6, axis='y')
    plt.legend()
    plt.savefig(os.path.join(figure_save_dir, f'{base_metric}_changes_due_to_{strategy}.pdf'),
                bbox_inches='tight')


# The below was inside visualize_resampling_effects.py right after obtain_results(), with
# (number_of_easy_classes, alpha_values) being the output of visualize_resampling_results.
""" 
# the alpha value (in samples_per_class[alpha] below) doesn't matter as long as its valid
class_order = np.argsort(samples_per_class[1])
for base_metric in ['Precision', 'Recall']:
    for strategy in results[base_metric].keys():
        plot_all_metrics_sorted(results, class_order, base_metric, number_of_easy_classes, strategy,
                                alpha_values, self.dataset_name, self.figure_save_dir)
        print(f'{"-" * 70}\n\tResults of {strategy} for {base_metric}:\n{"-" * 70}')
        plot_metric_changes(results, class_order, base_metric, number_of_easy_classes, strategy, alpha_values,
                            self.num_classes, self.dataset_name, self.figure_save_dir)
"""