"""Module that contains a few useful functions."""

from collections import defaultdict
import os
import pickle
import random
import re
from typing import Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns

from config import DEVICE, ROOT
from neural_networks import ResNet18LowRes


def set_reproducibility(seed: int = 42):
    """Crucial for ensuring code reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = False
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.deterministic = True


def get_latest_model_index(save_dir: str, num_epochs: int, max_dataset_count: int) -> List[int]:
    """Find the latest model index from saved files in the save directory. This makes it easier to add more models
    to the ensemble, as we don't have to retrain from scratch."""
    max_indices = defaultdict(lambda: -1)
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            match = re.search(rf'dataset_(\d+)_model_(\d+)_epoch_{num_epochs}\.pth$', filename)
            if match:
                dataset_idx = int(match.group(1))
                model_idx = int(match.group(2))
                max_indices[dataset_idx] = max(max_indices[dataset_idx], model_idx)
    return [max_indices[i] for i in range(max_dataset_count)]


def load_results(path: str):
    """Load results."""
    with open(path, 'rb') as file:
        return pickle.load(file)


def load_hardness_estimates(data_cleanliness: str, dataset_name: str) -> Dict[Tuple[int, int], Dict[str, List[float]]]:
    """Load hardness estimates."""
    path = os.path.join(ROOT, f'Results/{data_cleanliness}{dataset_name}', 'hardness_estimates.pkl')
    hardness_estimates = load_results(path)
    return hardness_estimates


def compute_sample_allocation_after_resampling(hardness_scores: List[float], labels: List[int], num_classes: int,
                                               num_training_samples: int, hardness_estimator: str,
                                               pruning_rate: int = 0, alpha: int = 1
                                               ) -> Tuple[List[int], Dict[int, List[float]]]:
    """Compute number of samples per class after hardness-based resampling according to hardness_scores."""
    hardnesses_by_class = {class_id: [] for class_id in range(num_classes)}

    # Divide the hardness estimates into classes.
    print(len(labels), len(hardness_scores))
    for i, label in enumerate(labels):
        hardnesses_by_class[label].append(hardness_scores[i])

    # Compute average hardness of each class.
    means_hardness_by_class = {class_id: np.mean(vals) for class_id, vals in hardnesses_by_class.items()}

    # Add offset in case some classes have negative hardness values to not get nonsensical resampling ratios.
    if min(means_hardness_by_class.values()) < 0:
        offset = -min(means_hardness_by_class.values())
        for class_id in range(num_classes):
            means_hardness_by_class[class_id] += offset

    # Compute the resampling ratios for each class.
    if hardness_estimator in ['AUM', 'Confidence', 'iAUM', 'iConfidence']:
        hardness_ratios = {class_id: 1 / float(val) for class_id, val in means_hardness_by_class.items()}
    else:
        hardness_ratios = {class_id: float(val) for class_id, val in means_hardness_by_class.items()}
    ratios = {class_id: class_hardness / sum(hardness_ratios.values())
              for class_id, class_hardness in hardness_ratios.items()}

    # Compute the amount of samples per class after resampling.
    samples_per_class = [int(round((1 - pruning_rate / 100) * ratio * num_training_samples))
                         for class_id, ratio in ratios.items()]

    # Tailor the degree of the introduces data imbalance (only applicable if alpha is larger than 1).
    if alpha > 1:
        average_sample_count = int(np.mean(samples_per_class))
        for class_id in range(num_classes):
            absolute_difference = abs(samples_per_class[class_id] - average_sample_count)
            if samples_per_class[class_id] > average_sample_count:
                samples_per_class[class_id] = average_sample_count + int(alpha * absolute_difference)
            else:
                samples_per_class[class_id] = average_sample_count - int(alpha * absolute_difference)

    return samples_per_class, hardnesses_by_class


def restructure_hardness_dictionary(
        hardness_estimates: Dict[Tuple[int, int], Dict[str, List[float]]]
) -> Dict[str, List[List[float]]]:
    """Used to change the structure of hardness_estimates (put the hardness estimator as the first key and model_idx as
    the second key)."""
    out = defaultdict(list)
    for _, estimates_by_estimator_name in hardness_estimates.items():
        for metric_name, estimates in estimates_by_estimator_name.items():
            out[metric_name].append(estimates)
    return dict(out)

def evaluate_ensembles(
        ensembles: List[List[str]],
        test_loader: DataLoader,
        class_index: int,
        num_classes: int,
        strategies: Union[str, Tuple[str, str]],
        degree: int,
        results: Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]]
):
    """Used to compute the scores necessary to produce visualizations - Tp, Tn, Fp, Fn, Precision, Recall, MCC, and F1.
    """
    for dataset_idx in range(len(ensembles)):
        for model_idx, model_path in enumerate(ensembles[dataset_idx]):
            model_state = torch.load(model_path)
            true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
            model = ResNet18LowRes(num_classes)
            model.load_state_dict(model_state)
            model = model.to(DEVICE)
            model.eval()

            with torch.no_grad():
                for images, labels, _ in test_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, predicted = outputs.max(1)

                    for pred, label in zip(predicted, labels):
                        if label == class_index:
                            if pred.item() == class_index:
                                true_positives += 1
                            else:
                                false_negatives += 1
                        else:
                            if pred.item() == class_index:
                                false_positives += 1
                            else:
                                true_negatives += 1

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            F1 = 2 * (precision * recall) / (precision + recall)  # noqa
            MCC_numerator = true_positives * true_negatives - false_positives * false_negatives  # noqa
            MCC_denominator = ((true_positives + false_positives) * (true_positives + false_negatives) *  # noqa
                               (true_negatives + false_positives) * (true_negatives + false_negatives)) ** 0.5
            MCC = MCC_numerator / MCC_denominator  # noqa

            for (metric_name, metric_results) in [('Tp', true_positives), ('Tn', true_negatives), ('Recall', recall),
                                                  ('Fp', false_positives), ('Fn', false_negatives), ('F1', F1),
                                                  ('MCC', MCC), ('Precision', precision)]:
                results[metric_name][strategies][degree][class_index][dataset_idx][model_idx] = metric_results


def obtain_results(save_dir: str, num_classes: int, test_loader: DataLoader, file_name: str,
                   models: Dict[Union[str, Tuple[str, str]], Dict[int, List[List[str]]]],
                   ) -> Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]]:
    """Load the results if they have been computed before, or use evaluate_ensembles to compute them."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(float))))))
    if os.path.exists(os.path.join(save_dir, file_name)):
        print('Loading pre-computed ensemble results.')
        with open(os.path.join(save_dir, file_name), 'rb') as f:
            results = pickle.load(f)
    else:
        for class_index in tqdm(range(num_classes), desc='Iterating through classes'):
            for strategies, ensembles_across_degrees in models.items():
                for degree, ensembles in ensembles_across_degrees.items():
                    evaluate_ensembles(ensembles, test_loader, class_index, num_classes, strategies, degree, results)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, file_name), "wb") as file:
            # noinspection PyTypeChecker
            pickle.dump(results, file)
    return results

def compute_fairness_metrics(
        results: Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]],
        samples_per_class: List[int],
        resampling_or_pruning_strategies: Union[List[str], List[Tuple[str, str]]],
        num_classes: int,
        max_ensemble_size: int
) -> Dict[str, Dict[str, Dict[str, Dict[int, Dict[int, float]]]]]:
    """This method computes the fairness metrics to better determine whether applying hardness-based resampling (both
    on pruned data in experiment2.py and the full data in experiment3.py) brings meaningful improvement to fairness."""
    fairness_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for base_metric in ['Recall', 'F1', 'MCC', 'Precision']:
        for strategy in resampling_or_pruning_strategies:
            for degree in sorted(results['Recall'][strategy].keys()):
                for ensemble_size in range(1, max_ensemble_size + 1):
                    recall_values = []
                    for class_id in range(num_classes):
                        # Mean recall over ensemble_size models for a given class
                        mean_recall = np.mean([
                            results['Recall'][strategy][degree][class_id][dataset_idx][model_idx]
                            for dataset_idx in range(ensemble_size)
                            for model_idx in range(ensemble_size)
                        ])
                        recall_values.append(mean_recall)

                    max_min_gap = max(recall_values) - min(recall_values)
                    std_dev = np.std(recall_values)
                    cv = float(np.std(recall_values)) / float(np.mean(recall_values))
                    quant_diff = float(np.percentile(recall_values, 90)) - float(np.percentile(recall_values, 10))
                    hard_class_recalls = [recall_values[cls] for cls in samples_per_class
                                          if samples_per_class[cls] > np.mean(samples_per_class)]
                    easy_class_recalls = [recall_values[cls] for cls in samples_per_class
                                          if samples_per_class[cls] <= np.mean(samples_per_class)]
                    hard_easy = abs(float(np.mean(hard_class_recalls)) - float(np.mean(easy_class_recalls)))
                    base_values = [np.mean([results['Recall'][strategy][0][class_id][dataset_idx][model_idx]
                                            for dataset_idx in range(ensemble_size)
                                            for model_idx in range(ensemble_size)])
                                   for class_id in range(num_classes)]
                    avg_change = float(np.mean(base_values)) - float(np.mean(recall_values))

                    for (fairness_metric, metric_results) in [('max_min_gap', max_min_gap), ('std_dev', std_dev),
                                                              ('cv', cv), ('quant_diff', quant_diff),
                                                              ('hard_easy', hard_easy), ('avg_change', avg_change)]:
                        fairness_results[base_metric][strategy][fairness_metric][degree][ensemble_size] = metric_results

    return fairness_results


def generate_fairness_table(fairness_results: Dict[str, Dict[str, Dict[str, Dict[int, Dict[int, float]]]]],
                            save_path: str, max_ensemble_size: int, task: str):
    """This produces a table to visualize the fairness gains from various hardness-based resampling variants."""
    for base_metric in ['Recall', 'F1', 'MCC', 'Precision']:
        rows = []
        for strategy, metrics_dict in fairness_results[base_metric].items():
            for degree in metrics_dict['max_min_gap']:
                row = {
                    f'{task} Strategy': strategy,
                    f'{task} Degree': degree,
                    'max_min_gap': metrics_dict['max_min_gap'][degree][max_ensemble_size],
                    'std_dev': metrics_dict['std_dev'][degree][max_ensemble_size],
                    'cv': metrics_dict['cv'][degree][max_ensemble_size],
                    'quant_diff': metrics_dict['quant_diff'][degree][max_ensemble_size],
                    'hard_easy': metrics_dict['hard_easy'][degree][max_ensemble_size],
                    'avg_change': metrics_dict['avg_change'][degree][max_ensemble_size]
                }
                rows.append(row)
        df = pd.DataFrame(rows)

        def quantize(s):
            """Helper function that improves clarity of fairness table."""
            if task == 'pruning':
                return [f'{v:.4f}' for v in s]
            else:
                min_val = s.min()
                return [f'\\textbf{{{v:.4f}}}' if v == min_val else f'{v:.4f}' for v in s]

        styled_df = df.copy()
        for col in ['max_min_gap', 'std_dev', 'cv', 'quant_diff', 'hard_easy', 'avg_change']:
            styled_df[col] = quantize(df[col])
        df.to_csv(os.path.join(save_path, f"{task}_{base_metric}_fairness_results.csv"), index=False)
        latex_str = styled_df.to_latex(index=False, escape=False)
        with open(os.path.join(save_path, f"{task}_{base_metric}_fairness_results.tex"), "w") as f:
            f.write(latex_str)


def plot_fairness_stability(fairness_results: Dict[str, Dict[str, Dict[str, Dict[int, Dict[int, float]]]]],
                            figure_save_dir: str):
    """This method visualizes the fairness_results using a heatmap. It's practical when the number of degrees is
    higher than one (we performed experiment3.py for more than one alpha value or experiment2.py for more than one
    pruning_rate). It's main focus is to determine if the ensemble size is adequate (are fairness_results stable with
    respect to the ensemble size or do we need to train more models or use more datasets)."""
    for base_metric in fairness_results.keys():
        for strategy in fairness_results[base_metric].keys():
            for fairness_metric in fairness_results[base_metric][strategy].keys():
                # Extract data into a 2D matrix: rows = degree of resampling/pruning, cols = ensemble sizes
                degrees = sorted(fairness_results[base_metric][strategy][fairness_metric].keys())
                ensemble_sizes = sorted(next(iter(
                    fairness_results[base_metric][strategy][fairness_metric].values()
                )).keys())

                heatmap_data = np.array([
                    [fairness_results[base_metric][strategy][fairness_metric][degree][size]
                     for size in ensemble_sizes]
                    for degree in degrees
                ])

                plt.figure(figsize=(10, 6))
                sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis",
                            xticklabels=ensemble_sizes, yticklabels=[f"{d}%" for d in degrees])
                plt.title(f"{fairness_metric.replace('_', ' ').title()} Heatmap ({strategy} based on "
                          f"{base_metric})")
                plt.xlabel("Ensemble Size")
                plt.ylabel("Degree of Resampling/Pruning")
                filename = f"{fairness_metric}_stability_{strategy}_based_on_{base_metric}.pdf"
                plt.tight_layout()
                plt.savefig(os.path.join(figure_save_dir, filename))
                plt.close()


def plot_fairness_dual_axis(fairness_results: Dict[str, Dict[str, Dict[str, Dict[int, Dict[int, float]]]]],
                            figure_save_dir: str, task: str):
    """This visualization compares the fairness across different task strategies. The idea is to make the findings from
     the fairness table more visually appealing."""
    colors = matplotlib.colormaps["tab10"]
    for base_metric in fairness_results.keys():
        strategies = list(fairness_results[base_metric].keys())
        for fairness_metric in fairness_results[base_metric][strategies[0]]:
            degrees = sorted(fairness_results[base_metric][strategies[0]][fairness_metric].keys())
            x_labels = [f"{d}" for d in degrees]
            fig, ax = plt.subplots(figsize=(10, 6))

            for i, (strategy) in enumerate(strategies):
                mean_values = [
                    np.mean(list(fairness_results[base_metric][strategy][fairness_metric][degree].values()))
                    for degree in degrees
                ]
                std_values = [
                    np.std(list(fairness_results[base_metric][strategy][fairness_metric][degree].values()))
                    for degree in degrees
                ]

                ax.plot(x_labels, mean_values, label=f"{strategy}", color=colors(i))
                ax.fill_between(x_labels, np.array(mean_values) - np.array(std_values),
                                np.array(mean_values) + np.array(std_values),  alpha=0.2, color=colors(i))

            ax.set_xlabel("Pruning Rate")
            ax.set_ylabel("Quant Diff")
            ax.tick_params(axis='y')
            lines_1, labels_1 = ax.get_legend_handles_labels()
            ax.legend(lines_1, labels_1, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

            plt.title(f"{fairness_metric} based on {base_metric} during {task}")
            fig.tight_layout()
            filename = f"{fairness_metric}_based_on_{base_metric}_during_{task}.pdf"
            plt.savefig(os.path.join(figure_save_dir, filename))
            plt.close()
