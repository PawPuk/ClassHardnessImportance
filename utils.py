"""Module that contains a few useful functions."""

from collections import defaultdict
import dill
import os
import pickle
import random
import re
from typing import Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def compute_sample_allocation_after_resampling(instance_hardness_scores: List[float], labels: List[int],
                                               num_classes: int, num_training_samples: int, hardness_estimator: str,
                                               pruning_rate: int = 0,
                                               alpha: int = 1) -> Tuple[List[int], Dict[int, List[Tuple[int, float]]]]:
    """Compute number of samples per class after hardness-based resampling according to hardness_scores."""
    # Divide the instant-level hardness estimates into classes.
    hardness_sorted_by_class = {class_id: [] for class_id in range(num_classes)}
    for i, label in enumerate(labels):
        hardness_sorted_by_class[label].append((i, instance_hardness_scores[i]))

    # Compute (or extract) average hardness of each class
    means_hardness_by_class = {class_id: np.mean([score for _, score in entries])
                               for class_id, entries in hardness_sorted_by_class.items()}
    print('means_hardness_by_class', means_hardness_by_class)

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

    return samples_per_class, hardness_sorted_by_class


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


def copy_results_for_degree_zero(results, first_strategies, strategies, degree, class_index, ensembles):
    """Since the results are shared across strategies for degree 0 we do not want to recompute them but rather copy to
    save compute."""
    for dataset_idx in range(len(ensembles)):
        for model_idx, _ in enumerate(ensembles[dataset_idx]):
            for metric_name in results.keys():
                results[metric_name][strategies][degree][class_index][dataset_idx][model_idx] = \
                    results[metric_name][first_strategies][degree][class_index][dataset_idx][model_idx]


def evaluate_ensembles(
        ensembles: Dict[int, List[str]],
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
                results[metric_name][strategies][int(degree)][class_index][dataset_idx][model_idx] = metric_results


def defaultdict_to_dict(d):
    """Recursively convert defaultdicts to dicts at all depths."""
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):  # catch plain dicts too
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def obtain_results(save_dir: str, num_classes: int, test_loader: DataLoader, file_name: str,
                   models: Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, List[str]]]],
                   ) -> Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]]:
    """Load the results if they have been computed before, or use evaluate_ensembles to compute them."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(float))))))
    if os.path.exists(os.path.join(save_dir, file_name)):
        print('Loading pre-computed ensemble results.')
        with open(os.path.join(save_dir, file_name), 'rb') as f:
            results = dill.load(f)
    else:
        for class_index in tqdm(range(num_classes), desc='Iterating through classes'):
            for i, (strategies, ensembles_across_degrees) in enumerate(models.items()):
                for degree, ensembles in ensembles_across_degrees.items():
                    evaluate_ensembles(ensembles, test_loader, class_index, num_classes, strategies, degree, results)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, file_name), "wb") as file:
            dill.dump(results, file)
    print('The loaded results have keys:', results.keys())
    return defaultdict_to_dict(results)


def compute_fairness_metrics(
        results: Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[int, Dict[int, Dict[int, float]]]]]],
        samples_per_class: List[int],
        resampling_or_pruning_strategies: Union[List[str], List[Tuple[str, str]]],
        num_classes: int
) -> Dict[str, Dict[Union[str, Tuple[str, str]], Dict[str, Dict[int, Tuple[float, float, List[float]]]]]]:
    """This method computes the fairness metrics to better determine whether applying hardness-based resampling (both
    on pruned data in experiment2.py and the full data in experiment3.py) brings meaningful improvement to fairness."""
    fairness_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for metric_name in ['Recall', 'F1', 'MCC', 'Precision']:
        for strategy in resampling_or_pruning_strategies:
            for degree in sorted(results[metric_name][strategy].keys()):
                fairness_metrics_grid = {'max_min_gap': [], 'std_dev': [], 'quant_diff': [], 'mad': [], 'hard_easy': [],
                                         'avg_value': []}

                num_datasets = len(results[metric_name][strategy][degree][0])
                for dataset_idx in range(num_datasets):
                    num_models = len(results[metric_name][strategy][degree][0][dataset_idx])
                    for model_idx in range(num_models):
                        metric_values = [
                            results[metric_name][strategy][degree][class_id][dataset_idx][model_idx]
                            for class_id in range(num_classes)
                            if results[metric_name][strategy][degree][class_id][dataset_idx][model_idx] != 0.0
                        ]

                        max_min_gap = max(metric_values) - min(metric_values)
                        std_dev = np.std(metric_values)

                        median_val = np.median(metric_values)
                        mad = np.median(np.abs(metric_values - median_val))

                        k = 2 if num_classes == 10 else 10
                        upper = np.mean(np.sort(metric_values)[-k:])
                        lower = np.mean(np.sort(metric_values)[:k])
                        quant_diff = upper - lower

                        hard_class_recalls = [metric_values[cls] for cls in range(len(samples_per_class))
                                              if samples_per_class[cls] > np.mean(samples_per_class)]
                        easy_class_recalls = [metric_values[cls] for cls in range(len(samples_per_class))
                                              if samples_per_class[cls] <= np.mean(samples_per_class)]
                        hard_easy = float(np.mean(easy_class_recalls)) - float(np.mean(hard_class_recalls))

                        avg_value = np.mean(metric_values)

                        for (fairness_metric, metric_results) in [('max_min_gap', max_min_gap), ('mad', mad),
                                                                  ('std_dev', std_dev), ('quant_diff', quant_diff),
                                                                  ('hard_easy', hard_easy),
                                                                  ('avg_value', avg_value)]:
                            fairness_metrics_grid[fairness_metric].append(metric_results)

                for fairness_metric, values in fairness_metrics_grid.items():
                    fairness_results[metric_name][strategy][fairness_metric][degree] = \
                        (float(np.mean(values)), float(np.std(values)), values)

    return defaultdict_to_dict(fairness_results)


def plot_fairness_dual_axis(
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
            fig, ax = plt.subplots(figsize=(10, 6))

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

            plt.title(f"{fairness_metric} based on {base_metric} during {task}")
            fig.tight_layout()
            filename = f"{fairness_metric}_based_on_{base_metric}_during_{task}.pdf"
            plt.savefig(os.path.join(figure_save_dir, filename))
            plt.close()


def perform_paired_t_tests(
        fairness_results: Dict[
            str, Dict[Union[str, Tuple[str, str]], Dict[str, Dict[int, Tuple[float, float, List[float]]]]]
        ],
        task: str
) -> Dict[str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[str, Tuple[float, float]]]]]:
    """
    Perform paired t-tests comparing fairness metrics at degree=0 (baseline) with non-zero degrees.
    Uses raw fairness values stored in fairness_results.
    """
    test_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for metric_name, strategies in fairness_results.items():
        for strategy, metrics_dict in strategies.items():
            for fairness_metric, degree_dict in metrics_dict.items():
                for degree, entry in degree_dict.items():
                    if task == 'resampling':
                        baseline_entry = degree_dict[0]
                        baseline_values = np.array(baseline_entry[2])
                    else:
                        baseline_entry = strategies['none'][fairness_metric][degree]
                        baseline_values = np.array(baseline_entry[2])

                    if degree == 0:
                        continue
                    comparison = np.array(entry[2])

                    # Align lengths
                    n = min(len(baseline_values), len(comparison))
                    if fairness_metric == 'avg_value':
                        baseline_aligned = comparison[:n]
                        comparison_aligned = baseline_values[:n]
                    else:
                        baseline_aligned = baseline_values[:n]
                        comparison_aligned = comparison[:n]

                    # Paired t-test (H0: mean diff = 0, H1: baseline > comparison)
                    t_stat, p_value = ttest_rel(baseline_aligned, comparison_aligned)

                    test_results[metric_name][strategy][degree][fairness_metric] = (float(t_stat), float(p_value))

    return defaultdict_to_dict(test_results)


def generate_t_test_table_for_fairness_metrics_for_pruning(
        fairness_t_test_results: Dict[
            str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[str, Tuple[float, float]]]]
        ],
        save_path: str
):
    """Generate a LaTeX table for pruning results only for the fairness metrics"""

    metrics = ['Recall', 'Precision']
    strategies = ['random', 'SMOTE', 'holdout']
    fairness_metrics = ['max_min_gap', 'quant_diff', 'hard_easy', 'std_dev', 'mad']

    strategy_names = {
        'random': 'random oversampling',
        'SMOTE': 'SMOTE oversampling',
        'holdout': 'holdout oversampling'
    }

    acronym_map = {'max_min_gap': 'm', 'quant_diff': 'q', 'hard_easy': 'he', 'std_dev': '\\sigma', 'mad': 'MAD'}

    pruning_rates = sorted(next(iter(fairness_t_test_results.values()))[strategies[0]].keys())

    latex_lines = [
        "\\begin{table}[h!]", "\\centering",
        "\\caption{t-statistics for fairness metrics under different pruning strategies. "
        "Significant results $(p < 0.05)$ are \\textbf{bolded}.}",
        "\\begin{tabular}{lc|" + "cc|" * 2 + "cc}",
        "\\toprule",
        "\\makecell{Fairness\\\\metric} & \\makecell{Pruning\\\\rate} " +
        "".join(
            f"& \\multicolumn{{2}}{{c|}}{{{strategy_names[s]}}} "
            if s != strategies[-1]
            else f"& \\multicolumn{{2}}{{c}}{{{strategy_names[s]}}}"
            for s in strategies
        ) + " \\\\",
        " & & " + " & ".join(["Recall & Precision"] * len(strategies)) + " \\\\",
        "\\midrule"
    ]

    for fm in fairness_metrics:
        latex_lines.append(f"\\multirow{{{len(pruning_rates)}}}{{*}}{{$\\Delta_{{{acronym_map[fm]}}}$}}")
        for i, rate in enumerate(pruning_rates):
            row = [f" & {rate}"]
            for strategy in strategies:
                for metric in metrics:
                    t, p = fairness_t_test_results[metric][strategy][rate][fm]
                    row.append(f"& \\textbf{{{t:.2f}}}" if p < 0.05 else f"& {t:.2f}")
            latex_lines.append(" ".join(row) + " \\\\")
        latex_lines.append("\\midrule")

    latex_lines.extend(["\\end{tabular}", "\\end{table}"])

    filename = os.path.join(save_path, "t_test_fairness_for_pruning.tex")
    with open(filename, "w") as f:
        f.write("\n".join(latex_lines))


def generate_t_test_table_for_avg_values_for_pruning(
        fairness_t_test_results: Dict[
            str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[str, Tuple[float, float]]]]
        ],
        save_path: str
):
    """
    Generate a LaTeX table for pruning results:
    - avg_value only
    - Recall, Precision, and F1
    """

    metrics = ['Recall', 'Precision', 'F1']
    strategies = ['random', 'SMOTE', 'holdout']

    strategy_names = {
        'random': 'random oversampling',
        'SMOTE': 'SMOTE oversampling',
        'holdout': 'holdout oversampling'
    }

    pruning_rates = sorted(next(iter(fairness_t_test_results.values()))[strategies[0]].keys())

    latex_lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{t-statistics for $\\Delta_{avg}$ under pruning. "
        "Significant results $(p<0.05)$ are \\textbf{bolded}.}",
        "\\begin{tabular}{lc|" + "ccc|" * 2 + "ccc}",
        "\\toprule",
        "\\makecell{Metric} & \\makecell{Pruning\\\\rate} " +
        "".join(
            f"& \\multicolumn{{3}}{{c|}}{{{strategy_names[s]}}} "
            if s != strategies[-1]
            else f"& \\multicolumn{{3}}{{c}}{{{strategy_names[s]}}}"
            for s in strategies
        ) + " \\\\",
        " & & " + " & ".join(["Recall & Precision & F1"] * len(strategies)) + " \\\\",
        "\\midrule"
    ]

    for metric in metrics:
        latex_lines.append(f"\\multirow{{{len(pruning_rates)}}}{{*}}{{{metric}}}")
        for rate in pruning_rates:
            row = [f" & {rate}"]
            for strategy in strategies:
                t, p = fairness_t_test_results[metric][strategy][rate]['avg_value']
                row.append(f"& \\textbf{{{t:.2f}}}" if p < 0.05 else f"& {t:.2f}")
            latex_lines.append(" ".join(row) + " \\\\")
        latex_lines.append("\\midrule")

    latex_lines.extend(["\\end{tabular}", "\\end{table}"])

    filename = os.path.join(save_path, "t_test_avg_for_pruning.tex")
    with open(filename, "w") as f:
        f.write("\n".join(latex_lines))


def generate_t_test_table_for_fairness_metrics_for_resampling(
        fairness_t_test_results: Dict[
            str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[str, Tuple[float, float]]]]
        ],
        save_path: str
):
    """Generate a LaTeX table for all fairness metrics."""

    def preferred_sort(strategies):
        order = ['random', 'SMOTE', 'rEDM', 'aEDM', 'hEDM']
        return sorted(
            strategies,
            key=lambda s: order.index(s[0]) if s[0] in order else len(order)
        )

    sample_metric = next(iter(fairness_t_test_results.keys()))
    all_strategies = list(fairness_t_test_results[sample_metric].keys())
    strategies_easy = preferred_sort([s for s in all_strategies if s[1] == "easy"])

    fairness_metrics = [
        ("max_min_gap", "m"),
        ("quant_diff", "q"),
        ("hard_easy", "he"),
        ("std_dev", "\\sigma"),
        ("mad", "MAD"),
    ]

    alphas = sorted(fairness_t_test_results[sample_metric][strategies_easy[0]].keys())
    per_group = "c" * len(alphas)
    col_spec = "l l|" + ("{}|".format(per_group) * (len(fairness_metrics) - 1)) + "{}".format(per_group)

    latex_lines = [
        "\\begin{sidewaystable}[htbp]",
        "    \\centering",
        "    \\caption{t-statistics (two decimals) for fairness metrics "
        "Significant results $(p<0.05)$ are \\textbf{bold}.}",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "        \\toprule"
    ]

    header1 = "            Metric & Oversampling " + "".join(
        f"& \\multicolumn{{{len(alphas)}}}{{c|}}{{$\\Delta_{{{a}}}$}} "
        for _, a in fairness_metrics[:-1]
    ) + f"& \\multicolumn{{{len(alphas)}}}{{c}}{{$\\Delta_{{{fairness_metrics[-1][1]}}}$}} \\\\"

    header2 = "            & " + \
              " & ".join([""] + [f"$\\alpha={a}$" for _ in fairness_metrics for a in alphas]) + \
              " \\\\"

    latex_lines.extend([header1, header2, "        \\midrule"])

    main_metrics = ["Recall", "Precision"]

    def format_row(strategy, main_metric):
        row = [strategy[0]]
        for fm, _ in fairness_metrics:
            for a in alphas:
                try:
                    t_stat, p_val = fairness_t_test_results[main_metric][strategy][a][fm]
                    row.append(f"\\textbf{{{t_stat:.2f}}}" if p_val < 0.05 else f"{t_stat:.2f}")
                except KeyError:
                    row.append("--")
        return " & ".join(row) + " \\\\"

    for mi, metric in enumerate(main_metrics):
        first = strategies_easy[0]
        latex_lines.append(
            f"            \\multirow{{{len(strategies_easy)}}}{{*}}{{{metric}}} & "
            + format_row(first, metric)
        )
        for s in strategies_easy[1:]:
            latex_lines.append("            & " + format_row(s, metric))
        if mi < len(main_metrics) - 1:
            latex_lines.append("        \\midrule")

    latex_lines.extend(["        \\bottomrule", "    \\end{tabular}", "\\end{sidewaystable}"])

    filename = os.path.join(save_path, "t_test_fairness_for_resampling.tex")
    with open(filename, "w") as f:
        f.write("\n".join(latex_lines))

    return filename


def generate_t_test_table_for_avg_values_for_resampling(
        fairness_t_test_results: Dict[
            str, Dict[Union[str, Tuple[str, str]], Dict[int, Dict[str, Tuple[float, float]]]]
        ],
        save_path: str
):
    """Generate a LaTeX table for avg_value only. Includes Recall, Precision, and F1."""

    def preferred_sort(strategies):
        order = ['random', 'SMOTE', 'rEDM', 'aEDM', 'hEDM']
        return sorted(
            strategies,
            key=lambda s: order.index(s[0]) if s[0] in order else len(order)
        )

    sample_metric = next(iter(fairness_t_test_results.keys()))
    all_strategies = list(fairness_t_test_results[sample_metric].keys())
    strategies_easy = preferred_sort([s for s in all_strategies if s[1] == "easy"])

    alphas = sorted(fairness_t_test_results[sample_metric][strategies_easy[0]].keys())
    col_spec = "l l|" + "c" * len(alphas)

    latex_lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{t-statistics for $\\Delta_{avg}$ across metrics. "
        "Significant results $(p<0.05)$ are \\textbf{bold}.}",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "        \\toprule",
        "            Metric & Oversampling & " +
        " & ".join(f"$\\alpha={a}$" for a in alphas) + " \\\\",
        "        \\midrule"
    ]

    main_metrics = ["Recall", "Precision", "F1"]

    def format_row(strategy, main_metric):
        row = [strategy[0]]
        for a in alphas:
            try:
                t, p = fairness_t_test_results[main_metric][strategy][a]["avg_value"]
                row.append(f"\\textbf{{{t:.2f}}}" if p < 0.05 else f"{t:.2f}")
            except KeyError:
                row.append("--")
        return " & ".join(row) + " \\\\"

    for mi, metric in enumerate(main_metrics):
        first = strategies_easy[0]
        latex_lines.append(
            f"            \\multirow{{{len(strategies_easy)}}}{{*}}{{{metric}}} & "
            + format_row(first, metric)
        )
        for s in strategies_easy[1:]:
            latex_lines.append("            & " + format_row(s, metric))
        if mi < len(main_metrics) - 1:
            latex_lines.append("        \\midrule")

    latex_lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}"
    ])

    filename = os.path.join(save_path, "t_test_avg_for_resampling.tex")
    with open(filename, "w") as f:
        f.write("\n".join(latex_lines))

    return filename

