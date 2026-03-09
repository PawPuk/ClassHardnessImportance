import os
from typing import Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_fairness(
        fairness_results: Dict[
            str, Dict[Union[str, Tuple[str, str]], Dict[str, Dict[int, Tuple[float, float, List[float]]]]]
        ],
        figure_save_dir: str,
        task: str
):
    """This visualization compares the fairness across different task strategies. The idea is to make the findings from
     the fairness table more visually appealing."""
    colors = matplotlib.colormaps["tab10"]
    for base_metric in ['Precision', 'Recall', 'F1', 'MCC']:
        strategies = sorted(list(fairness_results[base_metric].keys()))
        for fairness_metric in fairness_results[base_metric][strategies[0]]:
            degrees = sorted(fairness_results[base_metric][strategies[0]][fairness_metric].keys())
            x_labels = [f"{d}" for d in degrees]
            fig, ax = matplotlib.pyplot.subplots(figsize=(10, 6))

            for i, (strategy) in enumerate(strategies):
                means = [fairness_results[base_metric][strategy][fairness_metric][degree][0] for degree in degrees]
                if isinstance(strategy, tuple):
                    strategy = strategy[0]
                ax.plot(x_labels, means, label=f"{strategy}", color=colors(i))

            x_label = "Pruning_rate" if task == "pruning" else "Alpha"
            ax.set_xlabel(x_label)
            ax.set_ylabel("Quant Diff")
            ax.tick_params(axis='y')
            lines_1, labels_1 = ax.get_legend_handles_labels()
            ax.legend(lines_1, labels_1, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
            ax.grid(True, alpha=0.6, linestyle='--')

            matplotlib.pyplot.title(f"{fairness_metric} based on {base_metric} during {task}")
            fig.tight_layout()
            filename = f"{fairness_metric}_based_on_{base_metric}_during_{task}.pdf"
            matplotlib.pyplot.savefig(os.path.join(figure_save_dir, filename))
            matplotlib.pyplot.close()


def visualize_resampling_results(
        sample_allocation: Dict[int, List[int]],
        num_classes: int,
        dataset_name: str,
        figure_save_dir: str
) -> Tuple[int, List[int]]:
    """Produces components of Figure 4.

    This visualization shows the effects of resampling. It clearly shows how many samples were removed from each
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
    ax.set_xticks(np.arange(0, num_classes + 1))
    ax.set_ylabel('Class-wise sample count after resampling')
    ax.set_title(dataset_name)
    ax.legend()
    fig.savefig(os.path.join(figure_save_dir, 'sorted_resampled_dataset.pdf'))

    number_of_easy_classes = sum([sample_allocation[alpha_values[0]][cls] <= avg_count
                                  for cls in range(len(sample_allocation[alpha_values[0]]))])
    print(f'Identified {number_of_easy_classes} easy classes in this dataset.')
    return number_of_easy_classes, alpha_values
