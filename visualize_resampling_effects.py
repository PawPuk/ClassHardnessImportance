import argparse
from collections import defaultdict
import dill as pickle
import os
from typing import Dict, List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config, ROOT
from data import load_dataset
from neural_networks import ResNet18LowRes
from utils import load_results


class ResamplingVisualizer:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        config = get_config(args.dataset_name)
        self.target_ensemble_size = config['robust_ensemble_size']
        self.num_classes = config['num_classes']
        self.num_training_samples = config['num_training_samples']
        self.num_test_samples = config['num_test_samples']
        self.num_epochs = config['num_epochs']

        self.figure_save_dir = os.path.join(ROOT, 'Figures/', dataset_name)
        self.hardness_save_dir = os.path.join(ROOT, f"Results/unclean{self.dataset_name}/")

        for save_dir in self.figure_save_dir, self.hardness_save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def load_models(self) -> Dict[Tuple[str, str, str], Dict[float, List[dict]]]:
        """
        Load all models for the specified dataset, organizing by oversampling and undersampling strategies,
        and differentiating between cleaned and uncleaned datasets.

        :return: Dictionary where keys are tuples (oversampling_strategy, undersampling_strategy, dataset_type)
                 and values are lists of model state dictionaries.
        """
        models_dir = os.path.join(ROOT, "Models")
        models_by_strategy = defaultdict(lambda: defaultdict(list))

        for root, dirs, files in os.walk(models_dir):
            if 'clp' in root or 'dlp' in root:
                continue
            if os.path.basename(root) == f"unclean{self.dataset_name}" and 'over_' in root:
                dataset_type = 'unclean'
            elif f"clean{self.dataset_name}" in root and 'over_' in root and '_under_' in root:
                dataset_type = 'clean'
            else:
                continue

            try:
                oversampling_strategy = root.split("over_")[1].split("_under_")[0]
                undersampling_strategy = root.split("_under_")[1].split("_alpha_")[0]
                alpha = root.split("_alpha_")[1].split("_hardness_")[0]
                key = (oversampling_strategy, undersampling_strategy, dataset_type)

                for file in files:
                    if file.endswith(".pth") and "_epoch_200" in file:
                        try:
                            # Below ensures we load only the specified models (below the index threshold)
                            model_index = int(file.split("_")[1])
                            if model_index >= self.target_ensemble_size:
                                continue

                            model_path = os.path.join(root, file)
                            model_state = torch.load(model_path)
                            models_by_strategy[key][alpha].append(model_state)

                        # Skip directories or files that don't match the expected pattern
                        except (IndexError, ValueError):
                            continue
                if len(models_by_strategy[key][alpha]) > 0:
                    print(f"Loaded {len(models_by_strategy[key][alpha])} models for strategies {key} & alpha {alpha}.")
            except (IndexError, ValueError):
                continue

        # Also load models trained on the full dataset (no resampling)
        for dataset_type in ['unclean', 'clean']:
            full_dataset_dir = os.path.join(models_dir, "none", f"{dataset_type}{self.dataset_name}")
            if os.path.exists(full_dataset_dir):
                key = ('none', 'none', dataset_type)
                models_by_strategy[key][1] = []
                for file in os.listdir(full_dataset_dir):
                    if file.endswith(".pth") and "_epoch_200" in file:
                        try:
                            model_index = int(file.split("_")[1])
                            if model_index >= self.target_ensemble_size:
                                continue

                            model_path = os.path.join(full_dataset_dir, file)
                            model_state = torch.load(model_path)
                            models_by_strategy[key][1].append(model_state)
                            print(f"Loaded model for full dataset ({dataset_type}): {model_path}")

                        except (IndexError, ValueError):
                            continue

        print([key for key in models_by_strategy.keys()])
        print(f"Loaded {len(models_by_strategy.keys())} ensembles for {self.dataset_name}.")
        return models_by_strategy

    def evaluate_ensemble(self, ensemble: List[dict], test_loader, class_index: int, strategies: Tuple[str, str, str],
                          alpha: float, results):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_Tp, total_Fp, total_Fn, total_Tn = 0, 0, 0, 0

        for model_state in ensemble:
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
                                total_Tp += 1
                            else:
                                total_Fn += 1
                        else:
                            if pred.item() == class_index:
                                total_Fp += 1
                            else:
                                total_Tn += 1

        # Average over ensemble
        results['Tp'][strategies][alpha][class_index] = total_Tp / len(ensemble)
        results['Fp'][strategies][alpha][class_index] = total_Fp / len(ensemble)
        results['Fn'][strategies][alpha][class_index] = total_Fn / len(ensemble)
        results['Tn'][strategies][alpha][class_index] = total_Tn / len(ensemble)

    @staticmethod
    def save_file(save_dir, filename, data):
        os.makedirs(save_dir, exist_ok=True)
        save_location = os.path.join(save_dir, filename)
        with open(save_location, "wb") as file:
            pickle.dump(data, file)

    def obtain_results(self, result_dir: str, models: Dict[Tuple[str, str, str], Dict[float, List[dict]]],
                       test_loader: DataLoader, results):
        if os.path.exists(os.path.join(result_dir, "resampling_results.pkl")):
            with open(os.path.join(result_dir, "resampling_results.pkl"), 'rb') as f:
                results = pickle.load(f)
        else:
            for class_index in tqdm(range(self.num_classes), desc='Iterating through classes'):
                for (over, under, cleanliness), ensembles in models.items():
                    for alpha, ensemble in ensembles.items():
                        self.evaluate_ensemble(ensemble, test_loader, class_index, (over, under, cleanliness),
                                               alpha, results)
            self.save_file(result_dir, "resampling_results.pkl", results)
        return results

    def load_class_distributions(self) -> Dict[float, Dict[int, int]]:
        class_distributions = {}
        for root, dirs, files in os.walk(self.hardness_save_dir):
            if 'CIFAR100' in root and self.dataset_name != 'CIFAR100':
                continue  # This is required as 'CIFAR10' string is also contained in 'CIFAR100'...
            if f"unclean{self.dataset_name}" not in root or 'alpha_' not in root:
                continue

            alpha = int(root.split('alpha_')[-1])
            for file in files:
                file_path = os.path.join(root, file)
                class_distributions[alpha] = load_results(file_path)
        print(f'Loaded class distribution files:\n\t{class_distributions}')
        return class_distributions

    def visualize_resampling_results(self, samples_per_class):
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        fig2, ax2 = plt.subplots(figsize=(8, 5))

        alpha_values = sorted(map(int, samples_per_class.keys()))
        print('alpha values - ', alpha_values, samples_per_class.keys())
        norm = plt.Normalize(min(alpha_values), max(alpha_values))

        avg_count = np.mean(list(samples_per_class[alpha_values[0]].values()))
        min_count = min(min(samples.values()) for samples in samples_per_class.values())
        max_count = max(max(samples.values()) for samples in samples_per_class.values())

        ax1.axhspan(min_count, avg_count - 0.5, color='green', alpha=0.15)
        ax1.axhline(y=avg_count, color='blue', linestyle='--', linewidth=2, label='α=0')
        ax1.axhspan(avg_count - 0.5, max_count, color='red', alpha=0.15)

        ax2.axhspan(min_count, avg_count - 0.5, color='green', alpha=0.15)
        ax2.axhline(y=avg_count, color='blue', linestyle='--', linewidth=2, label='α=0')
        ax2.axhspan(avg_count - 0.5, max_count, color='red', alpha=0.15)

        for alpha in samples_per_class.keys():
            class_counts = list(samples_per_class[alpha].values())

            x = np.arange(self.num_classes)

            grey_shade = cm.viridis(0.8 * norm(alpha))  # Scale to [0.5, 1] for visual clarity
            ax1.plot(x, class_counts, color=grey_shade, label=f'α={alpha}', linewidth=2)

            # Plot class distribution in sorted order
            sorted_indices = np.argsort(class_counts)
            sorted_counts = np.array(class_counts)[sorted_indices]
            ax2.plot(range(len(sorted_counts)), sorted_counts, color=grey_shade, label=f'α={alpha}', linewidth=2)

        ax1.set_xlabel('Classes')
        ax1.set_xticklabels([])
        ax1.set_xticks(np.arange(1, self.num_classes + 1))
        ax1.set_ylabel('Class-wise sample count after resampling')
        ax1.set_title('Class Distribution (Natural Order)')
        ax1.legend(self.dataset_name)
        fig1.savefig(os.path.join(self.figure_save_dir, 'resampled_dataset.pdf'))

        ax2.set_xlabel('Classes sorted based on hardness (hardest to the right)')
        ax2.set_xticklabels([])
        ax2.set_xticks(np.arange(1, self.num_classes + 1))
        ax2.set_ylabel('Class-wise sample count after resampling')
        ax2.set_title(self.dataset_name)
        ax2.legend()
        fig2.savefig(os.path.join(self.figure_save_dir, 'sorted_resampled_dataset.pdf'))

        number_of_easy_classes = sum([samples_per_class[alpha_values[0]][i] <= avg_count
                                      for i in samples_per_class[alpha_values[0]].keys()])
        print(f'Finished plotting and computing the numbers of easy classes for each alpha: {number_of_easy_classes}')
        return number_of_easy_classes, alpha_values

    def plot_all_accuracies_sorted(self, results, class_order, base_metric, number_of_easy_classes, strategy,
                                   alpha_values):
        base_strategy, base_alpha, value_changes = ('none', 'none', 'unclean'), 1, {}

        plt.figure(figsize=(12, 8))
        reordered_base_values = [results[base_metric][base_strategy][base_alpha][class_id] for class_id in class_order]
        plt.plot(range(len(class_order)), [v for v in reordered_base_values], color='black', linestyle='-',
                 linewidth=4, label="α=1")

        """norm = plt.Normalize(min(alpha_values), max(alpha_values))
        for alpha in alpha_values:
            if alpha > 0:
                grey_shade = cm.Greys(0.5 + 0.5 * norm(alpha))
                reordered_values = [results[base_metric][strategy][str(alpha)][class_id] for class_id in class_order]
                plt.plot(range(len(reordered_values)), reordered_values, color=grey_shade, lw=2, linestyle = '--',
                         label=f'"α={alpha}"')"""

        plt.axvspan(0, number_of_easy_classes - 0.5, color='green', alpha=0.15)
        plt.axvspan(number_of_easy_classes - 0.5, len(class_order), color='red', alpha=0.15)

        plt.xlabel("Classes sorted based on hardness (hardest to the right)", fontsize=12)
        plt.xticks(np.arange(0, len(class_order)), labels=[])
        plt.ylabel(f"Class-wise {base_metric} on balanced dataset", fontsize=12)
        plt.title(self.dataset_name, fontsize=14)
        plt.grid(True, alpha=0.6, axis='y')
        # plt.legend()
        plt.savefig(os.path.join(self.figure_save_dir, f'{base_metric}_{strategy}_effects.pdf'),
                    bbox_inches='tight')

    def plot_metric_changes(self, results, class_order, base_metric, number_of_easy_classes, strategy, alpha_values):
        base_strategy, base_alpha, value_changes = ('none', 'none', 'unclean'), 1, defaultdict(lambda: defaultdict(list))

        plt.figure(figsize=(12, 8))
        norm = plt.Normalize(min(alpha_values), max(alpha_values))

        for alpha, values in results[base_metric][strategy].items():
            if float(alpha) > 0:
                value_changes[strategy][alpha] = [values[class_id] -
                                                  results[base_metric][base_strategy][base_alpha][class_id]
                                                  for class_id in class_order]
                grey_shade = cm.viridis(0.8 * norm(float(alpha)))
                print(f'{alpha}, {strategy} - easy mean: '
                      f'{np.mean(value_changes[strategy][alpha][:number_of_easy_classes])}'
                      f', hard mean:  {np.mean(value_changes[strategy][alpha][number_of_easy_classes:])}, easy std: '
                      f'{np.std(value_changes[strategy][alpha][:number_of_easy_classes])}, hard std:'
                      f'{np.std(value_changes[strategy][alpha][number_of_easy_classes:])}')
                plt.axhline(y=0, color='black', linewidth=2)
                plt.plot(range(len(values)), value_changes[strategy][alpha], color=grey_shade,
                         linewidth=2, label=f"α={alpha}")

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
        results_dir = os.path.join(ROOT, "Results", self.dataset_name)
        models = self.load_models()
        _, _, test_loader, _ = load_dataset(self.dataset_name, False, False, True)
        # results[metric][(over, under, cleanliness)][alpha][class_id] -> int
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        results = self.obtain_results(results_dir, models, test_loader, results)
        for strategies in results['Tp'].keys():
            for alpha in results['Tp'][strategies]:
                for class_id in range(self.num_classes):
                    Tp = results['Tp'][strategies][alpha][class_id]
                    Fp = results['Fp'][strategies][alpha][class_id]
                    Fn = results['Fn'][strategies][alpha][class_id]
                    Tn = sum(self.num_test_samples) / (Tp + Fp + Fn)

                    precision = Tp / (Tp + Fp) if (Tp + Fp) > 0 else 0.0
                    recall = Tp / (Tp + Fn) if (Tp + Fn) > 0 else 0.0
                    accuracy = (Tp + Tn) / sum(self.num_test_samples)
                    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                    MCC_numerator = Tp * Tn - Fp * Fn
                    MCC_denominator = ((Tp + Fp) * (Tp + Fn) * (Tn + Fp) * (Tn + Fn)) ** 0.5
                    MCC = 0.0 if MCC_denominator == 0 else MCC_numerator / MCC_denominator

                    for (metric_name, metric_results) in [('F1', F1), ('MCC', MCC), ('Tn', Tn), ('Precision', precision),
                                                          ('Average Model Accuracy', accuracy), ('Recall', recall)]:
                        results[metric_name][strategies][alpha][class_id] = metric_results

        samples_per_class = self.load_class_distributions()
        number_of_easy_classes, alpha_values = self.visualize_resampling_results(samples_per_class)

        class_order = np.argsort(list(samples_per_class[1].values()))

        for base_metric in ['F1', 'MCC', 'Precision', 'Recall']:
            for strategy in [('none', 'easy', 'unclean'), ('random', 'none', 'unclean'), ('SMOTE', 'none', 'unclean'),
                             ('easy', 'none', 'unclean'), ('hard', 'none', 'unclean'), ('random', 'easy', 'unclean'),
                             ('SMOTE', 'easy', 'unclean'), ('easy', 'easy', 'unclean'), ('hard', 'easy', 'unclean')]:

                self.plot_all_accuracies_sorted(results, class_order, base_metric, number_of_easy_classes, strategy,
                                                alpha_values)
                print('-'*20, f'\n\tResults of {strategy} for {base_metric}:\n', '-'*20)
                self.plot_metric_changes(results, class_order, base_metric, number_of_easy_classes, strategy,
                                         alpha_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (e.g., 'CIFAR10')")

    args = parser.parse_args()
    ResamplingVisualizer(args.dataset_name).main()
