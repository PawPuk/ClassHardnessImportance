import argparse
from collections import Counter
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config, ROOT
from data import load_dataset
from neural_networks import ResNet18LowRes
from utils import get_latest_model_index, load_aum_results, load_forgetting_results


class Visualizer:
    def __init__(self, dataset_name, remove_noise):
        self.dataset_name = dataset_name
        self.data_cleanliness = 'clean' if remove_noise else 'unclean'

        config = get_config(dataset_name)
        self.num_classes = config['num_classes']
        self.num_epochs = config['num_epochs']
        self.num_samples = sum(config['num_training_samples'])
        self.model_dir = config['save_dir']
        self.save_epoch = config['save_epoch']

        self.results_save_dir = os.path.join(ROOT, 'Results/', f'{self.data_cleanliness}{dataset_name}')
        self.figures_save_dir = os.path.join(ROOT, 'Figures/', f'{self.data_cleanliness}{dataset_name}')
        for save_dir in [self.results_save_dir, self.figures_save_dir]:
            os.makedirs(save_dir, exist_ok=True)

        self.pruning_thresholds = np.arange(10, 100, 10)
        model_save_dir = os.path.join(config['save_dir'], 'none', f'{self.data_cleanliness}{dataset_name}')
        self.num_models = get_latest_model_index(model_save_dir, self.num_epochs) + 1

    def load_model(self, model_id: int) -> ResNet18LowRes:
        model = ResNet18LowRes(num_classes=self.num_classes).cuda()
        model_path = os.path.join(self.model_dir, 'none', f"{self.data_cleanliness}{args.dataset_name}",
                                  f'model_{model_id}_epoch_{self.save_epoch}.pth')

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            # This code can only be run if models were pretrained. If no pretrained models are found, throw error.
            raise Exception(f'Model {model_id} not found at epoch {self.save_epoch}.')

        return model

    def compute_el2n_and_confidence(self, model: ResNet18LowRes,
                                    dataloader: DataLoader) -> Tuple[List[float], List[float]]:
        model.eval()
        el2n_scores, confidences = [], []

        with torch.no_grad():
            for inputs, labels, _ in dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)

                softmax_outputs = F.softmax(outputs, dim=1)
                one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()
                l2_errors = torch.norm(softmax_outputs - one_hot_labels, dim=1)
                el2n_scores.extend(l2_errors.cpu().numpy())

                max_confidences = torch.max(softmax_outputs, dim=1).values
                confidences.extend(max_confidences.cpu().numpy())

        return el2n_scores, confidences

    def collect_el2n_and_confidence_scores(self, loader):
        all_el2n_scores, all_confidences = [], []

        for model_id in range(self.num_models):
            model = self.load_model(model_id)
            el2n_scores, confidences = self.compute_el2n_and_confidence(model, loader)
            all_el2n_scores.append(el2n_scores)
            all_confidences.append(confidences)

        return all_el2n_scores, all_confidences

    def save_el2n_scores(self, el2n_scores):
        with open(os.path.join(self.results_save_dir, 'el2n_scores.pkl'), 'wb') as file:
            pickle.dump(el2n_scores, file)

    def get_pruned_indices(self, el2n_scores: List[List[float]], aum_scores: List[List[float]],
                           forgetting_scores: List[List[float]],
                           confidence_scores: List[List[float]]) -> Dict[str, List[List[List[int]]]]:
        """Extract the indices of data samples that would have been pruned if a threshold (from self.pruning_thresholds)
        was applied to EL2N, AUM, and Forgetting hardness estimates computed using ensembles of various sizes. This
        allows us to measure the reliability of those hardness estimates (e.g., how many models in the ensemble they
        need to output consistent pruning indices).
        """
        results = {}

        for metric_name, metric_scores in tqdm([("el2n", el2n_scores), ("aum", aum_scores),
                                                ("forgetting", forgetting_scores), ("confidence", confidence_scores)],
                                               desc='Computing pruned indices.'):
            metric_scores = np.array(metric_scores)
            results[metric_name] = []

            for thresh in self.pruning_thresholds:
                prune_count = int((thresh / 100) * self.num_samples)
                metric_results = []

                for num_ensemble_models in range(1, self.num_models + 1):
                    # Compute the average hardness score of each sample as a function of the ensemble size
                    avg_hardness_scores = np.mean(metric_scores[:num_ensemble_models], axis=0)
                    # For AUM and confidence, hard samples have lower values (opposite for EL2N and forgetting).
                    if metric_name in ['aum', 'confidence']:
                        sorted_indices = np.argsort(-avg_hardness_scores)
                    else:
                        sorted_indices = np.argsort(avg_hardness_scores)
                    pruned_indices = sorted_indices[:prune_count]
                    metric_results.append(pruned_indices.tolist())
                results[metric_name].append(metric_results)
        return results

    def compute_and_visualize_stability_of_pruning(self, results):
        metric_names = list(results.keys())
        # TODO: Modify the below so that it's not hardcoded.
        vmin, vmax = 0, 100  # Ensure all heatmaps share the same scale

        for metric_name in tqdm(metric_names, desc='Computing and visualizing stability of pruning across metrics'):
            metric_results = results[metric_name]
            num_pruning_thresholds = len(metric_results)
            num_models = len(metric_results[0])
            stability_results = np.zeros((num_pruning_thresholds, num_models - 1))

            for i in range(num_pruning_thresholds):
                for j in range(num_models - 1):
                    set1 = set(metric_results[i][j])  # Pruned indices for ensemble with j models
                    set2 = set(metric_results[i][j + 1])  # Pruned indices for ensemble with j+1 models

                    changed = len(set2 - set1) / len(set1)
                    stability_results[i, j] = changed * 100

            # Format the annotation values
            def custom_format(val):
                if val >= 10:
                    return f"{int(round(val))}"
                elif round(val) == 10 and val >= 9.95:
                    return f"{int(round(val))}"
                elif 1 < val < 10:
                    return f"{val:.1f}"
                elif round(val) == 1 and val > 0.99:
                    return f"{val:.1f}"
                else:
                    return f"{round(val, 2):.2f}"[1:]

            # Create figure and plot the heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(stability_results, annot=True, fmt='.2f', cmap='coolwarm',
                        cbar_kws={'label': 'Jaccard Overlap'}, vmin=vmin, vmax=vmax)  # Set color scale

            # Adjust the annotation format
            for text in plt.gca().texts:
                text.set_text(custom_format(float(text.get_text())))

            plt.title(f'Change in Pruned Indices After Increasing Ensemble Size (%) Based on {metric_name.upper()}')
            plt.xlabel('Number of models in ensemble (j) during hardness estimation')
            plt.ylabel('Pruning threshold (%)')
            plt.xticks(np.arange(num_models - 1) + 0.5, np.arange(1, num_models))
            plt.yticks(np.arange(num_pruning_thresholds) + 0.5, np.arange(10, 100, 10))
            plt.savefig(os.path.join(self.figures_save_dir, f'pruning_stability_based_on_{metric_name}.pdf'))
            plt.close()

    def compute_overlap_of_pruned_indices_across_hardness_estimators(self, results):
        metric_names = list(results.keys())
        num_metrics = len(metric_names)

        plt.figure(figsize=(10, 6))
        for i in tqdm(range(num_metrics), desc='Comparing pruned indices across hardness estimators'):
            for j in range(i + 1, num_metrics):  # Only compute unique pairs.
                overlaps = []
                for thresh_idx, thresh in enumerate(self.pruning_thresholds):
                    set1 = set(results[metric_names[i]][thresh_idx][-1])  # Using the full ensemble.
                    set2 = set(results[metric_names[j]][thresh_idx][-1])
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    overlap = intersection / union if union > 0 else 0.0
                    overlaps.append(overlap)

                plt.plot(self.pruning_thresholds, overlaps, label=f"{metric_names[i]} vs {metric_names[j]}",
                         marker='o')

        # Customizing the plot.
        plt.xlabel("Pruning rate (% of samples removed)")
        plt.ylabel("Overlap percentage")
        plt.title(self.dataset_name)
        plt.legend(title="Metric pairs")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_save_dir, f'overlap_across_hardness_estimators.pdf'))

    def compute_effect_of_ensemble_size_on_resampling(self, el2n_scores, aum_scores, forgetting_scores,
                                                      confidence_scores, dataset):
        results = {}
        for metric_name, metric_scores in tqdm([("el2n", el2n_scores), ("aum", aum_scores),
                                                ("forgetting", forgetting_scores), ("confidences", confidence_scores)],
                                               desc='Computing effect of ensemble size on resampling.'):
            metric_scores = np.array(metric_scores)
            results[metric_name] = []
            for num_ensemble_models in range(1, self.num_models):
                curr_avg_hardness_scores = np.mean(metric_scores[:num_ensemble_models], axis=0)
                next_avg_hardness_scores = np.mean(metric_scores[:num_ensemble_models + 1], axis=0)
                curr_class_results = [[] for _ in range(self.num_classes)]
                next_class_results = [[] for _ in range(self.num_classes)]
                curr_class_hardness, next_class_hardness = {}, {}

                for i, (_, label, _) in enumerate(dataset):
                    curr_class_results[label].append(curr_avg_hardness_scores[i])
                    next_class_results[label].append(next_avg_hardness_scores[i])
                for label in range(self.num_classes):
                    # For AUM, high values indicate easier samples so inversion is necessary
                    if metric_name == 'aum':
                        curr_class_hardness[label] = 1 / np.mean(curr_class_results[label])
                        next_class_hardness[label] = 1 / np.mean(next_class_results[label])
                    else:
                        curr_class_hardness[label] = np.mean(curr_class_results[label])
                        next_class_hardness[label] = np.mean(next_class_results[label])

                curr_ratios = {class_id: curr_hardness / sum(curr_class_hardness.values())
                               for class_id, curr_hardness in curr_class_hardness.items()}
                next_ratios = {class_id: next_hardness / sum(next_class_hardness.values())
                               for class_id, next_hardness in next_class_hardness.items()}

                curr_samples_per_class = {class_id: int(round(curr_ratio * self.num_samples))
                                          for class_id, curr_ratio in curr_ratios.items()}
                next_samples_per_class = {class_id: int(round(next_ratio * self.num_samples))
                                          for class_id, next_ratio in next_ratios.items()}

                differences = [abs(next_samples_per_class[k] - curr_samples_per_class[k])
                               for k in range(self.num_classes)]
                relative_differences = [differences[k] / curr_samples_per_class[k] for k in range(self.num_classes)]
                results[metric_name].append((num_ensemble_models, differences, relative_differences))
        return results

    def visualize_stability_of_resampling(self, results):
        metrics = list(results.keys())
        ensemble_sizes = {metric: [entry[0] for entry in results[metric]] for metric in metrics}
        differences = {metric: [entry[1] for entry in results[metric]] for metric in metrics}
        relative_differences = {metric: [entry[2] for entry in results[metric]] for metric in metrics}

        def compute_stats(diff_list):
            max_vals = [np.max(diff) for diff in diff_list]
            min_vals = [np.min(diff) for diff in diff_list]
            avg_vals = [np.mean(diff) for diff in diff_list]
            return max_vals, min_vals, avg_vals

        for metric in metrics:
            max_diffs, min_diffs, avg_diffs = compute_stats(differences[metric])
            if metric == 'el2n':
                print(max_diffs, min_diffs, avg_diffs)
            plt.figure(figsize=(10, 6))
            plt.plot(ensemble_sizes[metric], max_diffs, label=f'Max Diff', linestyle='-', marker='o')
            plt.plot(ensemble_sizes[metric], min_diffs, label=f'Min Diff', linestyle='--', marker='x')
            plt.plot(ensemble_sizes[metric], avg_diffs, label=f'Avg Diff', linestyle=':', marker='s')
            plt.title(f"Based on {metric.upper()}")
            plt.xlabel('Number of Models (j) in Ensemble During Hardness Estimation')
            plt.xticks(range(1, 20, 2))
            plt.ylabel("Absolute Difference in Class-Wise SXample Count for Resampling after Adding a Model")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(self.figures_save_dir, f'absolute_differences_{metric}.pdf'))

            max_rel_diffs, min_rel_diffs, avg_rel_diffs = compute_stats(relative_differences[metric])
            plt.figure(figsize=(10, 6))
            plt.plot(ensemble_sizes[metric], max_rel_diffs, label=f'Max Rel Diff', linestyle='-', marker='o')
            plt.plot(ensemble_sizes[metric], min_rel_diffs, label=f'Min Rel Diff', linestyle='--', marker='x')
            plt.plot(ensemble_sizes[metric], avg_rel_diffs, label=f'Avg Rel Diff', linestyle=':', marker='s')
            plt.title(f"Based on {metric.upper()}")
            plt.xlabel('Number of Models (j) in Ensemble During Hardness Estimation')
            plt.xticks(range(1, 20, 2))
            plt.ylabel("Relative Difference in Class-Wise SXample Count for Resampling after Adding a Model")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(self.figures_save_dir, f'relative_differences_{metric}.pdf'))

    def main(self):
        training_loader, _, _, _ = load_dataset(args.dataset_name, self.data_cleanliness == 'clean', False, True)
        training_all_el2n_scores, training_all_confidences = self.collect_el2n_and_confidence_scores(training_loader)
        self.save_el2n_scores(training_all_el2n_scores)

        aum_scores = load_aum_results(self.data_cleanliness, self.dataset_name, self.num_epochs)
        forgetting_scores = load_forgetting_results(self.data_cleanliness, self.dataset_name)

        print('All of the below should have the same dimensions. Otherwise, there is something wrong with the code.')
        print(f'Shape of hardness estimated via AUM: {len(aum_scores)}, {len(aum_scores[0])}')
        print(f'Shape of hardness estimated via Forgetting: {len(forgetting_scores)}, {len(forgetting_scores[0])}')
        print(f'Shape of hardness estimated via EL2N: {len(training_all_el2n_scores)}, '
              f'{len(training_all_el2n_scores[0])}')
        print()

        # pruned_indices[metric_name][pruning_threshold][num_ensemble_models][pruned_indices]
        pruned_indices = self.get_pruned_indices(training_all_el2n_scores, aum_scores, forgetting_scores,
                                                 training_all_confidences)
        self.compute_and_visualize_stability_of_pruning(pruned_indices)
        self.compute_overlap_of_pruned_indices_across_hardness_estimators(pruned_indices)

        differences = self.compute_effect_of_ensemble_size_on_resampling(
            training_all_el2n_scores, aum_scores, forgetting_scores, training_all_confidences, training_loader.dataset)
        self.visualize_stability_of_resampling(differences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Hardness of Dataset Samples')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='Specify the dataset name (default: CIFAR10)')
    parser.add_argument('--remove_noise', action='store_true', help='Raise this flag to remove noise from the data.')
    args = parser.parse_args()

    Visualizer(args.dataset_name, args.remove_noise).main()
