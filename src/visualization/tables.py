"""Module that contains a few useful functions."""

import os
from typing import Dict, Tuple, Union


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
