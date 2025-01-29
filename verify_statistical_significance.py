import argparse
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_networks import ResNet18LowRes
from utils import get_config, load_dataset


class Visualizer:
    def __init__(self, dataset_name, remove_noise):
        config = get_config(args.dataset_name)
        self.num_classes = config['num_classes']
        self.num_epochs = config['num_epochs']
        self.num_samples = config['num_samples']
        self.model_dir = config['model_dir']
        self.save_epoch = config['save_epoch']
        self.data_cleanliness = 'clean' if args.remove_noise else 'unclean'

        self.results_save_dir = os.path.join('Results/', f"{self.data_cleanliness}{args.dataset_name}")
        self.figures_save_dir = os.path.join('Figures/', f'{self.data_cleanliness}{args.dataset_name}')
        self.hardness_save_dir = f"Results/{self.data_cleanliness}{args.dataset_name}/"
        for save_dir in [self.results_save_dir, self.figures_save_dir, self.hardness_save_dir]:
            os.makedirs(save_dir, exist_ok=True)

        self.thresholds = np.arange(10, 100, 10)

        # Modify the below to extract this from the folder (check how many models were trained)
        NUM_MODELS = get_config(args.dataset_name)['num_models']

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

    def compute_el2n(self, model: ResNet18LowRes, dataloader: DataLoader) -> List[float]:
        model.eval()
        el2n_scores = []

        with torch.no_grad():
            for inputs, labels, _ in dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)

                softmax_outputs = F.softmax(outputs, dim=1)
                one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()
                l2_errors = torch.norm(softmax_outputs - one_hot_labels, dim=1)
                el2n_scores.extend(l2_errors.cpu().numpy())

        return el2n_scores

    def collect_el2n_scores(self, loader):
        all_el2n_scores = []

        for model_id in range(self.num_models):
            model = self.load_model(model_id)
            el2n_scores = self.compute_el2n(model, loader)
            all_el2n_scores.append(el2n_scores)

        return all_el2n_scores

    def save_el2n_scores(self, el2n_scores):
        with open(os.path.join(self.results_save_dir, 'el2n_scores.pkl'), 'wb') as file:
            pickle.dump(el2n_scores, file)

    @staticmethod
    def load_results(path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def load_aum_results(self) -> List[List[float]]:
        """Loading the AUM results and changing their format to match that of other hardness estimators by summing over
        epochs."""
        aum_path = os.path.join(self.hardness_save_dir, 'AUM.pkl')
        aum_over_epochs_and_models = self.load_results(aum_path)

        #TODO: This is CURRENTLY required as train_ensemble.py wasn't initially working properly with denoised datasets.
        for model_idx, model_list in enumerate(aum_over_epochs_and_models):
            aum_over_epochs_and_models[model_idx] = [sample for sample in model_list if len(sample) > 0]

        aum_scores = [
            [
                sum(model_list[sample_idx][epoch_idx] for epoch_idx in range(self.num_epochs)) / self.num_epochs
                for sample_idx in range(len(aum_over_epochs_and_models[0]))
            ]
            for model_list in aum_over_epochs_and_models
        ]

        return aum_scores

    def load_forgetting_results(self, aum_scores) -> List[List[float]]:
        forgetting_path = os.path.join(self.hardness_save_dir, 'Forgetting.pkl')
        forgetting_scores = self.load_results(forgetting_path)

        # TODO: This is CURRENTLY required as train_ensemble.py wasn't initially working properly with denoised datasets.
        forgetting_scores = [model_list[:len(aum_scores[0])] for model_list in forgetting_scores]
        return forgetting_scores


    def get_pruned_indices(self, el2n_scores: List[List[float]], aum_scores: List[List[float]],
                           forgetting_scores: List[List[float]]) -> Dict[str, List[List[List[int]]]]:
        results = {}

        for metric_name, metric_scores in tqdm([("el2n", el2n_scores), ("aum", aum_scores),
                                                ("forgetting", forgetting_scores)], desc='Iterating through metrics.'):
            metric_scores = np.array(metric_scores)
            results[metric_name] = []

            for thresh in self.thresholds:
                prune_count = int((thresh / 100) * sum(self.num_samples))
                metric_results = []

                for num_ensemble_models in range(1, self.num_models + 1):
                    avg_hardness_scores = np.mean(metric_scores[:num_ensemble_models], axis=0)
                    # For AUM hard samples have lower values. For EL2N and forgetting the opposite is the case.
                    if metric_name == 'aum':
                        sorted_indices = np.argsort(-avg_hardness_scores)
                    else:
                        sorted_indices = np.argsort(avg_hardness_scores)
                    pruned_indices = sorted_indices[:prune_count]
                    metric_results.append(pruned_indices.tolist())
                results[metric_name].append(metric_results)
        return results

    def compute_and_visualize_stability_of_pruning(self, results):
        metric_names = list(results.keys())

        for metric_name in metric_names:
            metric_results = results[metric_name]
            num_thresholds = len(metric_results)
            num_models = len(metric_results[0])
            stability_results = np.zeros((num_thresholds, num_models))

            for i in range(num_thresholds):
                for j in range(num_models - 1):
                    set1 = set(metric_results[i][j])  # Pruned indices for ensemble with j models
                    set2 = set(metric_results[i][j + 1])  # Pruned indices for ensemble with j+1 models

                    changed = len(set2 - set1) / len(set1)
                    stability_results[i, j] = changed * 100

            plt.figure(figsize=(10, 6))
            sns.heatmap(stability_results, annot=True, cmap='coolwarm', cbar_kws={'label': 'Jaccard Overlap'})
            plt.title(f'Pruned Indices Change (%) After Adding a Model {metric_name.upper()}')
            plt.xlabel('Number of Models in Ensemble (j) Before Adding a Model')
            plt.ylabel('Pruning Threshold (%)')
            plt.xticks(np.arange(num_models) + 0.5, np.arange(1, num_models + 1))
            plt.yticks(np.arange(num_thresholds) + 0.5, np.arange(10, 100, 10))
            plt.savefig(os.path.join(self.figures_save_dir, f'pruning_stability_based_on_{metric_name}.pdf'))


    def pruned_indices_vs_hardness_estimator(self, results):
        metric_names = list(results.keys())
        num_metrics = len(metric_names)

        plt.figure(figsize=(10, 6))
        for i in range(num_metrics):
            for j in range(i + 1, num_metrics):  # Only compute unique pairs.
                overlaps = []
                for t, thresh in enumerate(self.thresholds):
                    set1 = set(results[metric_names[i]][t][-1])  # Using the full ensemble.
                    set2 = set(results[metric_names[j]][t][-1])
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    overlap = intersection / union if union > 0 else 0.0
                    overlaps.append(overlap)

                # Plot overlaps as a function of thresholds.
                plt.plot(self.thresholds, overlaps, label=f"{metric_names[i]} vs {metric_names[j]}", marker='o')

        # Customizing the plot.
        plt.xlabel("Pruning Threshold (%)")
        plt.ylabel("Overlap Percentage (%)")
        plt.title("Overlap of Pruned Sets Across Hardness Estimators")
        plt.legend(title="Metric Pairs")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_save_dir, f'overlap_across_hardness_estimators.pdf'))


    def compute_effect_of_ensemble_size_on_resampling(el2n_scores, aum_scores, forgetting_scores, dataset):
        num_models = len(el2n_scores)  # number of models in the ensemble
        results = {}
        for metric_name, metric_scores in tqdm([("el2n", el2n_scores), ("aum", aum_scores),
                                                ("forgetting", forgetting_scores)], desc='Iterating through metrics.'):
            metric_scores = np.array(metric_scores)
            results[metric_name] = []
            for num_ensemble_models in range(2, num_models + 1):
                prev_avg_hardness_scores = np.mean(metric_scores[:num_ensemble_models - 1], axis=0)
                curr_avg_hardness_scores = np.mean(metric_scores[:num_ensemble_models], axis=0)
                prev_class_results, curr_class_results = [[] for _ in range(NUM_CLASSES)], [[] for _ in range(NUM_CLASSES)]
                prev_ratios, curr_ratios = {}, {}

                for i, (_, label, _) in enumerate(dataset):
                    prev_class_results[label].append(prev_avg_hardness_scores[i])
                    curr_class_results[label].append(curr_avg_hardness_scores[i])
                for label in range(NUM_CLASSES):
                    prev_ratios[label] = np.mean(prev_class_results[label])
                    curr_ratios[label] = np.mean(curr_class_results[label])

                prev_normalized_ratios = {class_id: ratio / sum(prev_ratios.values())
                                          for class_id, ratio in prev_ratios.items()}
                curr_normalized_ratios = {class_id: ratio / sum(curr_ratios.values())
                                          for class_id, ratio in curr_ratios.items()}

                prev_samples_per_class = {class_id: int(round(prev_normalized_ratio * NUM_SAMPLES))
                                          for class_id, prev_normalized_ratio in prev_normalized_ratios.items()}
                curr_samples_per_class = {class_id: int(round(curr_normalized_ratio * NUM_SAMPLES))
                                          for class_id, curr_normalized_ratio in curr_normalized_ratios.items()}

                differences = [abs(curr_samples_per_class[l] - prev_samples_per_class[l]) for l in range(NUM_CLASSES)]
                relative_differences = [differences[l] / prev_samples_per_class[l] for l in range(NUM_CLASSES)]
                results[metric_name].append((num_ensemble_models, differences, relative_differences))
        return results


    def visualize_stability_of_resampling(results):

        metrics = list(results.keys())
        ensemble_sizes = {metric: [entry[0] for entry in results[metric]] for metric in metrics}
        differences = {metric: [entry[1] for entry in results[metric]] for metric in metrics}
        relative_differences = {metric: [entry[2] for entry in results[metric]] for metric in metrics}

        def compute_stats(diff_list):
            max_vals = [np.max(diff) for diff in diff_list]
            min_vals = [np.min(diff) for diff in diff_list]
            avg_vals = [np.mean(diff) for diff in diff_list]
            return max_vals, min_vals, avg_vals

        plt.figure(figsize=(15, 8))
        for metric in metrics:
            max_diffs, min_diffs, avg_diffs = compute_stats(differences[metric])
            plt.plot(ensemble_sizes[metric], max_diffs, label=f'{metric.upper()} Max Diff', linestyle='-', marker='o')
            plt.plot(ensemble_sizes[metric], min_diffs, label=f'{metric.upper()} Min Diff', linestyle='--', marker='x')
            plt.plot(ensemble_sizes[metric], avg_diffs, label=f'{metric.upper()} Avg Diff', linestyle=':', marker='s')
        plt.title("Absolute Differences Across Ensemble Sizes for Hardness Estimators")
        plt.xlabel('Ensemble Size')
        plt.ylabel("Absolute Difference")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(FIGURES_SAVE_DIR, f'absolute_differences_across_ensemble_sizes.pdf'))

        plt.figure(figsize=(15, 8))
        for metric in metrics:
            max_rel_diffs, min_rel_diffs, avg_rel_diffs = compute_stats(relative_differences[metric])
            plt.plot(ensemble_sizes[metric], max_rel_diffs, label=f'{metric.upper()} Max Rel Diff', linestyle='-',
                     marker='o')
            plt.plot(ensemble_sizes[metric], min_rel_diffs, label=f'{metric.upper()} Min Rel Diff', linestyle='--',
                     marker='x')
            plt.plot(ensemble_sizes[metric], avg_rel_diffs, label=f'{metric.upper()} Avg Rel Diff', linestyle=':',
                     marker='s')
        plt.title("Relative Differences Across Ensemble Sizes for Hardness Estimators")
        plt.xlabel('Ensemble Size')
        plt.ylabel("Relative Difference")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(FIGURES_SAVE_DIR, f'relative_differences_across_ensemble_sizes.pdf'))

        # I produce also the below Figures to allow for more clear analysis of the results (metrics have different values)
        for metric in metrics:
            max_diffs, min_diffs, avg_diffs = compute_stats(differences[metric])
            plt.figure(figsize=(10, 6))
            plt.plot(ensemble_sizes[metric], max_diffs, label=f'Max Diff', linestyle='-', marker='o')
            plt.plot(ensemble_sizes[metric], min_diffs, label=f'Min Diff', linestyle='--', marker='x')
            plt.plot(ensemble_sizes[metric], avg_diffs, label=f'Avg Diff', linestyle=':', marker='s')
            plt.title(f"Absolute Differences for {metric.upper()}")
            plt.xlabel('Ensemble Size')
            plt.ylabel("Absolute Difference")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(FIGURES_SAVE_DIR, f'absolute_differences_{metric}.pdf'))

            max_rel_diffs, min_rel_diffs, avg_rel_diffs = compute_stats(relative_differences[metric])
            plt.figure(figsize=(10, 6))
            plt.plot(ensemble_sizes[metric], max_rel_diffs, label=f'Max Rel Diff', linestyle='-', marker='o')
            plt.plot(ensemble_sizes[metric], min_rel_diffs, label=f'Min Rel Diff', linestyle='--', marker='x')
            plt.plot(ensemble_sizes[metric], avg_rel_diffs, label=f'Avg Rel Diff', linestyle=':', marker='s')
            plt.title(f"Relative Differences for {metric.upper()}")
            plt.xlabel('Ensemble Size')
            plt.ylabel("Relative Difference")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(FIGURES_SAVE_DIR, f'relative_differences_{metric}.pdf'))

    def main(self):
        training_loader, _, _, _ = load_dataset(args.dataset_name, args.remove_noise, False)
        training_all_el2n_scores = self.collect_el2n_scores(training_loader)
        self.save_el2n_scores(training_all_el2n_scores)

        aum_scores = self.load_aum_results()
        forgetting_scores = self.load_forgetting_results(aum_scores)

        print(len(aum_scores), len(aum_scores[0]))
        print(len(forgetting_scores), len(forgetting_scores[0]))
        print(len(training_all_el2n_scores), len(training_all_el2n_scores[0]))
        print()

        print('Measuring the stability of data pruning with respect to different hardness estimators.')
        pruned_indices = self.get_pruned_indices(training_all_el2n_scores, aum_scores, forgetting_scores)
        self.compute_and_visualize_stability_of_pruning(pruned_indices)
        self.pruned_indices_vs_hardness_estimator(pruned_indices)

        differences = compute_effect_of_ensemble_size_on_resampling(
            training_all_el2n_scores, AUM_scores, forgetting_scores, training_loader.dataset)
        visualize_stability_of_resampling(differences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Hardness of Dataset Samples')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='Specify the dataset name (default: CIFAR10)')
    parser.add_argument('--remove_noise', action='store_true', help='Raise this flag to remove noise from the data.')
    args = parser.parse_args()

    Visualizer(args.dataset_name, args.remove_noise).main()
