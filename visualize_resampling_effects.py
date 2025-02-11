import argparse
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from data import load_dataset
from neural_networks import ResNet18LowRes
from utils import load_results


def load_models(dataset_name: str) -> Dict[Tuple[str, str, str], List[dict]]:
    """
    Load all models for the specified dataset, organizing by oversampling and undersampling strategies,
    and differentiating between cleaned and uncleaned datasets.

    :param dataset_name: Name of the dataset (e.g., 'CIFAR10').
    :return: Dictionary where keys are tuples (oversampling_strategy, undersampling_strategy, dataset_type)
             and values are lists of model state dictionaries.
    """
    models_dir, models_by_strategy = "Models", {}
    target_ensemble_size = config['robust_ensemble_size']

    for root, dirs, files in os.walk(models_dir):
        if 'CIFAR100' in root and dataset_name != 'CIFAR100':
            continue  # This is required as 'CIFAR10' string is also contained in 'CIFAR100'...
        elif f"unclean{dataset_name}" in root and 'over_' in root and '_under_' in root:
            dataset_type = 'unclean'
        elif f"clean{dataset_name}" in root and 'over_' in root and '_under_' in root:
            dataset_type = 'clean'
        else:
            continue

        try:
            oversampling_strategy = root.split("over_")[1].split("_under_")[0]
            undersampling_strategy = root.split("_under_")[1].split("_size_")[0]
            key = (oversampling_strategy, undersampling_strategy, dataset_type)
            models_by_strategy.setdefault(key, [])

            for file in files:
                if file.endswith(".pth") and "_epoch_200" in file:
                    try:
                        # Below ensures we load only the specified models (below the index threshold)
                        model_index = int(file.split("_")[1])
                        if model_index >= target_ensemble_size:
                            continue

                        model_path = os.path.join(root, file)
                        model_state = torch.load(model_path)
                        models_by_strategy[key].append(model_state)

                    # Skip directories or files that don't match the expected pattern
                    except (IndexError, ValueError):
                        continue
            if len(models_by_strategy[key]) > 0:
                print(f"Loaded {len(models_by_strategy[key])} models for strategies {key}.")
        except (IndexError, ValueError):
            continue

    # Also load models trained on the full dataset (no resampling)
    for dataset_type in ['unclean', 'clean']:
        full_dataset_dir = os.path.join(models_dir, "none", f"{dataset_type}{dataset_name}")
        if os.path.exists(full_dataset_dir):
            key = ('none', 'none', dataset_type)
            models_by_strategy[key] = []
            for file in os.listdir(full_dataset_dir):
                if file.endswith(".pth") and "_epoch_200" in file:
                    try:
                        model_index = int(file.split("_")[1])
                        if model_index >= target_ensemble_size:
                            continue

                        model_path = os.path.join(full_dataset_dir, file)
                        model_state = torch.load(model_path)
                        models_by_strategy[key].append(model_state)
                        print(f"Loaded model for full dataset ({dataset_type}): {model_path}")

                    except (IndexError, ValueError):
                        continue

    print([key for key in models_by_strategy.keys()])
    print(f"Loaded {len(models_by_strategy.keys())} ensembles for {dataset_name}.")
    return models_by_strategy


def evaluate_ensemble(ensemble: List[dict], test_loader, class_index: int, strategies: Tuple[str, str, str], results):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_Tp, total_Fp, total_Fn, total_Tn = 0, 0, 0, 0

    for model_state in ensemble:
        model = ResNet18LowRes(num_classes)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)

                for pred, label in zip(predicted, labels):
                    if label.item() == class_index:
                        if pred.item() == class_index:
                            total_Tp += 1
                        else:
                            total_Fn += 1
                    else:
                        if pred.item() == class_index:
                            total_Fp += 1
                        else:
                            total_Tn += 1

    # Average over ensemble
    results['Tp'][strategies][class_index] = total_Tp / len(ensemble)
    results['Fp'][strategies][class_index] = total_Fp / len(ensemble)
    results['Fn'][strategies][class_index] = total_Fn / len(ensemble)
    results['Tn'][strategies][class_index] = total_Tn / len(ensemble)


def save_file(save_dir, filename, data):
    os.makedirs(save_dir, exist_ok=True)
    save_location = os.path.join(save_dir, filename)
    with open(save_location, "wb") as file:
        pickle.dump(data, file)


def obtain_results(result_dir: str, models: Dict[Tuple[str, str, str], List[dict]], test_loader: DataLoader):
    if os.path.exists(os.path.join(result_dir, "resampling_results.pkl")):
        with open(os.path.join(result_dir, "resampling_results.pkl"), 'rb') as f:
            results = pickle.load(f)
    else:
        results = {'Tp': {}, 'Fn': {}, 'Fp': {}, 'Tn': {}}

        for class_index in tqdm(range(num_classes), desc='Iterating through classes'):
            for (over, under, cleanliness), ensemble in models.items():
                if class_index == 0:
                    for metric in ['Tp', 'Fp', 'Fn', 'Tn']:
                        results[metric][(over, under, cleanliness)] = {class_id: 0 for class_id in range(num_classes)}
                evaluate_ensemble(ensemble, test_loader, class_index, (over, under, cleanliness), results)
        save_file(result_dir, "resampling_results.pkl", results)
    return results


def get_class_order(results, base_strategy, base_metric):
    """Determine class order based on average accuracies for the specified base strategy."""
    accuracies = results[base_metric][base_strategy]
    sorted_classes = sorted(accuracies.keys(), key=lambda k: accuracies[k])
    return sorted_classes


def plot_lsvc_accuracies(accuracy_dict, custom_order):
    mean_accuracies = []

    for class_id in accuracy_dict.keys():
        accuracies = [acc[1] for acc in accuracy_dict[class_id]]
        mean_accuracies.append(np.mean(accuracies))

    # Ensure custom order is applied correctly
    sorted_means = np.array(mean_accuracies)[custom_order]

    # Sequential x-axis for sorted plot
    x_sorted = np.arange(len(sorted_means))

    # Plot results
    plt.figure(figsize=(7, 5))
    plt.plot(x_sorted, sorted_means, linestyle='-')
    plt.xlabel("Sorted Class Index (Based on ResNet performance)")
    plt.ylabel("Accuracy")
    plt.title("Class-wise Accuracies of LSVCs")
    plt.grid()
    plt.savefig(os.path.join(figure_save_dir, 'resampling_effects.pdf'))


def plot_all_accuracies_sorted(results, class_order, base_metric, number_of_easy_samples, target_strategies, title):
    base_strategy, value_changes = ('none', 'none', 'unclean'), {}

    # Plot for all strategies using line segments
    plt.figure(figsize=(12, 8))
    color_palette = plt.cm.tab10(np.linspace(0, 1, len(results[base_metric])))

    # We plot the results for the base strategies using black solid line for clarity
    reordered_base_values = [results[base_metric][base_strategy][class_id] for class_id in class_order]
    plt.plot(range(len(class_order)), [v for v in reordered_base_values],
             color='black', linestyle='-', linewidth=4, label="none-none-unclean")

    for idx, (strategy, values) in enumerate(results[base_metric].items()):
        if strategy not in target_strategies:
            continue
        reordered_values = [values[class_id] for class_id in class_order]

        plt.plot(range(len(reordered_values)), reordered_values, color=color_palette[idx], lw=2)

    plt.axvspan(0, number_of_easy_samples - 0.5, color='green', alpha=0.15)
    plt.axvline(x=number_of_easy_samples - 0.5, color='blue', linestyle='--', linewidth=2)
    plt.axvspan(number_of_easy_samples - 0.5, len(class_order), color='red', alpha=0.15)

    plt.xlabel("Class (Sorted by Base Strategy)", fontsize=12)
    plt.xticks([])
    plt.ylabel(f"{base_metric} (%)", fontsize=12)
    plt.title(f"{base_metric} by Class for All Strategies (Sorted by Base Strategy)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(figure_save_dir, f'{base_metric}_{title}_effects.pdf'), bbox_inches='tight')


def plot_metric_changes(results, class_order, base_metric, number_of_easy_samples, target_strategies, title):
    base_strategy, value_changes = ('none', 'none', 'unclean'), {}
    for idx, (strategy, values) in enumerate(results[base_metric].items()):
        if strategy not in target_strategies:
            continue
        value_changes[strategy] = [values[class_id] - results[base_metric][base_strategy][class_id]
                                   for class_id in class_order]

    color_palette = plt.cm.tab10(np.linspace(0, 1, len(results[base_metric])))
    plt.figure(figsize=(12, 8))
    for strategy_idx, (strategy, values) in enumerate(value_changes.items()):
        if strategy == ('easy', 'easy', 'unclean') and base_metric == 'Recall':
            print(f'\n\n\n{number_of_easy_samples}{values}\n\n\n')
        print(f'{strategy} - easy mean: {np.mean(value_changes[strategy][:number_of_easy_samples])}'
              f', hard mean:  {np.mean(value_changes[strategy][number_of_easy_samples:])}, easy std: '
              f'{np.std(value_changes[strategy][:number_of_easy_samples])}, hard std:'
              f'{np.std(value_changes[strategy][number_of_easy_samples:])}')
        plt.axhline(y=0, color='black', linewidth=2)
        plt.plot(range(len(values)), values, color=color_palette[strategy_idx], linestyle='--', linewidth=2)

    plt.axvspan(0, number_of_easy_samples - 0.5, color='green', alpha=0.15)
    plt.axvline(x=number_of_easy_samples - 0.5, color='blue', linestyle='--', linewidth=2)
    plt.axvspan(number_of_easy_samples - 0.5, len(class_order), color='red', alpha=0.15)

    plt.xlabel("Class (Sorted by Base Strategy)", fontsize=12)
    plt.ylabel(f"{base_metric} Change (%)", fontsize=12)
    plt.title(f"{base_metric} Change Relative to Base Strategy (Sorted by Base Strategy)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(figure_save_dir, f'{base_metric}_changes_due_to_{title}.pdf'), bbox_inches='tight')


def main(dataset_name):
    number_of_easy_samples = 6 if dataset_name == 'CIFAR10' else 59
    hardness_save_dir = f"Results/unclean{dataset_name}/"
    samples_per_class = load_results(os.path.join(hardness_save_dir, 'samples_per_class.pkl'))
    samples_per_class = [samples_per_class[idx] for idx in samples_per_class.keys()]
    class_order = np.argsort(samples_per_class)

    results_dir = os.path.join("Results", dataset_name)
    models = load_models(dataset_name)
    _, training_dataset, test_loader, _ = load_dataset(dataset_name, False, False, False)

    # Evaluate ensemble performance
    results = obtain_results(results_dir, models, test_loader)

    for metric_name in ['F1', 'MCC', 'Tn', 'Average Model Accuracy', 'Precision', 'Recall']:
        results[metric_name] = {}
    for strategies in results['Tp'].keys():
        for metric_name in ['F1', 'MCC', 'Tn', 'Average Model Accuracy', 'Precision', 'Recall']:
            results[metric_name][strategies] = {}
        for class_id in range(num_classes):
            Tp = results['Tp'][strategies][class_id]
            Fp = results['Fp'][strategies][class_id]
            Fn = results['Fn'][strategies][class_id]
            Tn = sum(num_test_samples) / (Tp + Fp + Fn)

            precision = Tp / (Tp + Fp) if (Tp + Fp) > 0 else 0.0
            recall = Tp / (Tp + Fn) if (Tp + Fn) > 0 else 0.0
            accuracy = (Tp + Tn) / sum(num_test_samples)
            F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            MCC_numerator = Tp * Tn - Fp * Fn
            MCC_denominator = ((Tp + Fp) * (Tp + Fn) * (Tn + Fp) * (Tn + Fn)) ** 0.5
            MCC = 0.0 if MCC_denominator == 0 else MCC_numerator / MCC_denominator

            for (metric_name, metric_results) in [('F1', F1), ('MCC', MCC), ('Tn', Tn), ('Precision', precision),
                                                  ('Average Model Accuracy', accuracy), ('Recall', recall)]:
                results[metric_name][strategies][class_id] = metric_results

    for base_metric in ['F1', 'MCC', 'Tn', 'Average Model Accuracy', 'Precision', 'Recall', 'Tp', 'Fp', 'Fn']:

        """class_overlap_estimates = np.load('ovo_accuracies.npy', allow_pickle=True).item()
        mean_accuracies = []
        for class_id in class_overlap_estimates.keys():
            accuracies = [acc[1] for acc in class_overlap_estimates[class_id]]
            mean_accuracies.append(np.mean(accuracies))
        plot_lsvc_accuracies(class_overlap_estimates, class_order)"""
        plot_all_accuracies_sorted(results, class_order, base_metric, number_of_easy_samples,
                                   [('none', 'easy', 'unclean')], 'undersample')
        plot_all_accuracies_sorted(results, class_order, base_metric, number_of_easy_samples,
                                   [('random', 'none', 'unclean'), ('easy', 'none', 'unclean'),
                                    ('hard', 'none', 'unclean'), ('SMOTE', 'none', 'unclean')], 'oversample')
        plot_all_accuracies_sorted(results, class_order, base_metric, number_of_easy_samples,
                                   [], 'resample')

        print('-'*20, f'\n\tResults of undersampling for {base_metric}:\n', '-'*20)
        plot_metric_changes(results, class_order, base_metric, number_of_easy_samples,
                            [('none', 'easy', 'unclean')], 'undersample')
        print('-'*20, f'\n\tResults of oversampling for {base_metric}:\n', '-'*20)
        plot_metric_changes(results, class_order, base_metric, number_of_easy_samples,
                            [('random', 'none', 'unclean'), ('easy', 'none', 'unclean'),
                             ('hard', 'none', 'unclean'), ('SMOTE', 'none', 'unclean')], 'oversample')
        print('-'*20, f'\n\tResults of resampling for {base_metric}:\n', '-'*20)
        plot_metric_changes(results, class_order, base_metric, number_of_easy_samples,
                            [('easy', 'easy', 'unclean')], 'resample')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (e.g., 'CIFAR10')")

    args = parser.parse_args()

    figure_save_dir = os.path.join('Figures/', args.dataset_name)
    config = get_config(args.dataset_name)
    num_classes = config['num_classes']
    num_training_samples = config['num_training_samples']
    num_test_samples = config['num_test_samples']
    num_epochs = config['num_epochs']
    os.makedirs(figure_save_dir, exist_ok=True)
    main(args.dataset_name)
