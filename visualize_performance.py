import argparse
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_networks import ResNet18LowRes
from data import load_dataset
from config import get_config


class PerformanceVisualizer:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        config = get_config(args.dataset_name)
        self.num_classes = config['num_classes']
        self.num_epochs = config['num_epochs']
        self.num_training_samples = config['num_training_samples']
        self.num_test_samples = config['num_test_samples']

        self.figure_save_dir = os.path.join('Figures/', args.dataset_name)
        os.makedirs(self.figure_save_dir, exist_ok=True)

    def load_models(self) -> Dict[str, Dict[int, List[dict]]]:
        models_dir = "/mnt/parscratch/users/acq21pp/ClassHardnessImportance/Models"
        models_by_rate = {'fclp': {}, 'dlp': {}}

        for pruning_strategy in ['fclp', 'dlp']:
            # Walk through each folder in the Models directory
            for root, dirs, files in os.walk(models_dir):
                # Check if the path matches the specified pruning strategy and dataset name
                if f"{pruning_strategy}" in root and f"{self.dataset_name}" in root:
                    pruning_rate = int(root.split(pruning_strategy)[1].split("/")[0])
                    models_by_rate[pruning_strategy].setdefault(pruning_rate, [])
                    for file in files:
                        if file.endswith(".pth") and "_epoch_200" in file:
                            model_path = os.path.join(root, file)
                            model_state = torch.load(model_path)
                            models_by_rate[pruning_strategy][pruning_rate].append(model_state)

            # Load models trained on the full dataset (no pruning)
            full_dataset_dir = os.path.join(models_dir, "none", f'unclean{self.dataset_name}')
            if os.path.exists(full_dataset_dir):
                models_by_rate[pruning_strategy][0] = []  # Use `0` to represent models without pruning
                for file in os.listdir(full_dataset_dir):
                    if file.endswith(".pth") and "_epoch_200" in file:
                        model_path = os.path.join(full_dataset_dir, file)
                        model_state = torch.load(model_path)
                        models_by_rate[pruning_strategy][0].append(model_state)
            print(f"Models loaded by pruning rate for {pruning_strategy} on {self.dataset_name}")
        return models_by_rate

    def compute_pruned_percentage(self, models_by_rate: [str, Dict[int, List[dict]]]) -> Dict[str, Dict[
                                                                                              int, List[float]]]:
        pruned_percentages = {}
        for pruning_strategy in ['fclp', 'dlp']:
            pruning_rates = models_by_rate[pruning_strategy].keys()
            pruned_percentages[pruning_strategy] = {pruning_rate: [] for pruning_rate in pruning_rates}

            # Iterate over each pruning rate in models_by_rate
            for pruning_rate in pruning_rates:
                if pruning_rate != 0:
                    pkl_path = os.path.join("/mnt/parscratch/users/acq21pp/ClassHardnessImportance/Results",
                                            pruning_strategy + str(pruning_rate), self.dataset_name,
                                            f"class_level_sample_counts.pkl")
                    with open(pkl_path, "rb") as file:
                        class_level_sample_counts = pickle.load(file)
                    remaining_data_count = class_level_sample_counts[pruning_strategy][pruning_rate]
                    for c in range(self.num_classes):
                        pruned_percentage = 100.0 * (self.num_training_samples[c] - remaining_data_count[c]) / \
                                            self.num_training_samples[c]
                        pruned_percentages[pruning_strategy][pruning_rate].append(pruned_percentage)
                else:
                    pruned_percentages[pruning_strategy][0] = [0.0 for _ in range(self.num_classes)]

        return pruned_percentages

    def plot_pruned_percentages(self, pruned_percentages: Dict[int, List[float]]):
        plt.figure(figsize=(10, 6))

        pruning_thresholds = sorted(pruned_percentages.keys())
        for class_idx in range(self.num_classes):
            class_pruning_values = [pruned_percentages[threshold][class_idx] for threshold in pruning_thresholds]
            plt.plot(pruning_thresholds, class_pruning_values, marker='o')

        plt.xlabel("Dataset-Level Pruning Rate (%)")
        plt.ylabel("Class-Level Pruning Percentage (%)")
        plt.grid(True)

        plt.savefig(os.path.join(self.figure_save_dir, f'class_vs_dataset_pruning_values.pdf'))

    def evaluate_block(self, ensemble: List[dict], test_loader, class_index: int, pruning_rate: int, results,
                       pruning_strategy: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize confusion matrix components
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
        results[pruning_strategy]['Tp'][pruning_rate][class_index] = total_Tp / len(ensemble)
        results[pruning_strategy]['Fp'][pruning_rate][class_index] = total_Fp / len(ensemble)
        results[pruning_strategy]['Fn'][pruning_rate][class_index] = total_Fn / len(ensemble)
        results[pruning_strategy]['Tn'][pruning_rate][class_index] = total_Tn / len(ensemble)

    def evaluate_ensemble(self, models_by_rate: Dict[str, Dict[int, List[dict]]], test_loader: DataLoader,
                          results: Dict[str, Dict[str, Dict]]):
        for pruning_strategy in ['fclp', 'dlp']:
            for pruning_rate, ensemble in tqdm(models_by_rate[pruning_strategy].items(),
                                               desc=f'Iterating over pruning rates ({pruning_strategy}).'):
                for metric_name in ['Tp', 'Fp', 'Fn', 'Tn']:
                    results[pruning_strategy][metric_name][pruning_rate] = {class_id: 0
                                                                            for class_id in range(self.num_classes)}
                for class_index in range(self.num_classes):
                    self.evaluate_block(ensemble, test_loader, class_index, pruning_rate, results, pruning_strategy)

    def plot_class_level_results(self, results, pruned_percentages):
        pruning_rates = sorted(results['fclp']['Average Model Accuracy'].keys())

        for metric_name in results['fclp'].keys():
            fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey=True)
            fig.suptitle(f"Class-Level  {metric_name} Across Pruning Rates", fontsize=16)
            axes = axes.flatten()

            for class_id in range(self.num_classes):
                metric_value_fclp = [results['fclp'][metric_name][p][class_id] for p in pruning_rates]
                metric_value_dlp = [results['dlp'][metric_name][p][class_id] for p in pruning_rates]

                pruning_fclp = [pruned_percentages['fclp'][p][class_id] for p in pruning_rates]
                pruning_dlp = [pruned_percentages['dlp'][p][class_id] for p in pruning_rates]

                ax = axes[class_id]
                ax.plot(pruning_fclp, metric_value_fclp, marker='o', linestyle='-', color='blue', label='FCLP')
                ax.set_xlabel("Pruning % (clp)", color='blue')
                ax.set_xlim(0, 100)
                ax.tick_params(axis='x', labelcolor='blue')

                ax2 = ax.twiny()
                ax2.plot(pruning_dlp, metric_value_dlp, marker='s', linestyle='--', color='red', label='DLP')
                ax2.set_xlabel("Pruning % (dlp)", color='red')
                ax2.set_xlim(0, 100)  # Fix x-axis range for DLP
                ax2.tick_params(axis='x', labelcolor='red')

                ax.set_ylabel(f"{metric_name}")
                ax.set_title(f"Class {class_id}")
                ax.grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
            plt.savefig(os.path.join(self.figure_save_dir, f'Class_Level_{metric_name}.pdf'))
            plt.close()

    def compare_fclp_with_dlp(self, results):
        pruning_rates = sorted(results['fclp']['Average Model Accuracy'].keys())

        for metric_name in results['fclp'].keys():
            avg_metric_fclp = [np.mean([results['fclp'][metric_name][p][class_id]
                                        for class_id in range(self.num_classes)]) for p in pruning_rates]
            std_metric_fclp = [np.std([results['fclp'][metric_name][p][class_id]
                                       for class_id in range(self.num_classes)]) for p in pruning_rates]

            avg_metric_dlp = [np.mean([results['dlp'][metric_name][p][class_id]
                                       for class_id in range(self.num_classes)]) for p in pruning_rates]
            std_metric_dlp = [np.std([results['dlp'][metric_name][p][class_id]
                                      for class_id in range(self.num_classes)]) for p in pruning_rates]

            # Create a figure
            plt.figure(figsize=(8, 6))

            # Plot FCLP (Blue Line)
            plt.plot(pruning_rates, avg_metric_fclp, marker='o', linestyle='-', color='blue', label='fclp')
            plt.fill_between(pruning_rates,
                             np.array(avg_metric_fclp) - np.array(std_metric_fclp),
                             np.array(avg_metric_fclp) + np.array(std_metric_fclp),
                             color='blue', alpha=0.2)  # Shaded region for std

            # Plot DLP (Red Line)
            plt.plot(pruning_rates, avg_metric_dlp, marker='s', linestyle='--', color='red', label='dlp')
            plt.fill_between(pruning_rates,
                             np.array(avg_metric_dlp) - np.array(std_metric_dlp),
                             np.array(avg_metric_dlp) + np.array(std_metric_dlp),
                             color='red', alpha=0.2)  # Shaded region for std

            # Labels & Legend
            plt.xlabel("Pruning Percentage", fontsize=12)
            plt.ylabel(f"Average {metric_name}", fontsize=12)
            plt.xlim(0, 100)  # Ensure pruning percentage is between 0 and 100
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.savefig(os.path.join(self.figure_save_dir, f'{metric_name}_fclp_vs_dlp.pdf'))

    @staticmethod
    def save_file(save_dir, filename, data):
        os.makedirs(save_dir, exist_ok=True)
        save_location = os.path.join(save_dir, filename)
        with open(save_location, "wb") as file:
            pickle.dump(data, file)

    def main(self):
        result_dir = os.path.join("/mnt/parscratch/users/acq21pp/ClassHardnessImportance/Results/", self.dataset_name)
        models = self.load_models()
        pruned_percentages = self.compute_pruned_percentage(models)
        if self.num_classes == 10:
            self.plot_pruned_percentages(pruned_percentages['dlp'])
        _, _, test_loader, _ = load_dataset(self.dataset_name, False, False, False)

        # Evaluate ensemble performance
        if os.path.exists(os.path.join(result_dir, "ensemble_results.pkl")):
            print('Loading pre-computed ensemble results.')
            with open(os.path.join(result_dir, "ensemble_results.pkl"), 'rb') as f:
                results = pickle.load(f)
        else:
            print('Evaluating performance of ensembles trained on variously pruned dataset.')

            results = {}
            for pruning_strategy in ['fclp', 'dlp']:
                results[pruning_strategy] = {}
                for metric_name in ['Tp', 'Fn', 'Fp', 'Tn']:
                    results[pruning_strategy][metric_name] = {}

            self.evaluate_ensemble(models, test_loader, results)
            self.save_file(result_dir, "ensemble_results.pkl", results)

        for pruning_strategy in ['fclp', 'dlp']:
            for metric_name in ['F1', 'MCC', 'Tn', 'Average Model Accuracy', 'Precision', 'Recall']:
                results[pruning_strategy][metric_name] = {}
            for pruning_rate in results['fclp']['Tp'].keys():
                for metric_name in ['F1', 'MCC', 'Tn', 'Average Model Accuracy', 'Precision', 'Recall']:
                    results[pruning_strategy][metric_name][pruning_rate] = {}
                for class_id in range(self.num_classes):
                    Tp = results[pruning_strategy]['Tp'][pruning_rate][class_id]
                    Fp = results[pruning_strategy]['Fp'][pruning_rate][class_id]
                    Fn = results[pruning_strategy]['Fn'][pruning_rate][class_id]
                    Tn = sum(self.num_test_samples) / (Tp + Fp + Fn)

                    precision = Tp / (Tp + Fp) if (Tp + Fp) > 0 else 0.0
                    recall = Tp / (Tp + Fn) if (Tp + Fn) > 0 else 0.0
                    accuracy = Tp + Tn / sum(self.num_test_samples)
                    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                    MCC_numerator = Tp * Tn - Fp * Fn
                    MCC_denominator = ((Tp + Fp) * (Tp + Fn) * (Tn + Fp) * (Tn + Fn)) ** 0.5
                    MCC = 0.0 if MCC_denominator == 0 else MCC_numerator / MCC_denominator

                    results[pruning_strategy]['F1'][pruning_rate][class_id] = F1
                    results[pruning_strategy]['MCC'][pruning_rate][class_id] = MCC
                    results[pruning_strategy]['Tn'][pruning_rate][class_id] = Tn
                    results[pruning_strategy]['Average Model Accuracy'][pruning_rate][class_id] = accuracy
                    results[pruning_strategy]['Precision'][pruning_rate][class_id] = precision
                    results[pruning_strategy]['Recall'][pruning_rate][class_id] = recall

        if self.num_classes == 10:
            self.plot_class_level_results(results, pruned_percentages)
        self.compare_fclp_with_dlp(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., 'CIFAR10')")
    args = parser.parse_args()
    PerformanceVisualizer(args.dataset_name).main()
