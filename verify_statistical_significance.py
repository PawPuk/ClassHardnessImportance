import argparse
from itertools import combinations
import os
from typing import Dict, List

from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from config import get_config, ROOT
from data import AugmentedSubset, load_dataset
from neural_networks import ResNet18LowRes
from utils import get_latest_model_index, load_hardness_estimates


class Visualizer:
    def __init__(self, dataset_name, remove_noise):
        self.dataset_name = dataset_name
        self.data_cleanliness = 'clean' if remove_noise else 'unclean'

        config = get_config(dataset_name)
        self.num_classes = config['num_classes']
        self.num_epochs = config['num_epochs']
        self.num_training_samples = sum(config['num_training_samples'])
        self.model_dir = config['save_dir']
        self.save_epoch = config['save_epoch']

        self.results_save_dir = os.path.join(ROOT, 'Results/', f'{self.data_cleanliness}{dataset_name}')
        self.figures_save_dir = os.path.join(ROOT, 'Figures/', f'{self.data_cleanliness}{dataset_name}')
        for save_dir in [self.results_save_dir, self.figures_save_dir]:
            os.makedirs(save_dir, exist_ok=True)

        self.pruning_thresholds = np.array([10, 20, 30, 40, 50])
        model_save_dir = os.path.join(config['save_dir'], 'none', f'{self.data_cleanliness}{dataset_name}')
        self.num_models = get_latest_model_index(model_save_dir, self.num_epochs) + 1

    def load_model(self, model_id: int, probe=False) -> ResNet18LowRes:
        model = ResNet18LowRes(num_classes=self.num_classes).cuda()
        epoch = self.save_epoch if probe else self.num_epochs
        model_path = os.path.join(self.model_dir, 'none', f"{self.data_cleanliness}{args.dataset_name}",
                                  f'model_{model_id}_epoch_{epoch}.pth')

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            # This code can only be run if models were pretrained. If no pretrained models are found, throw error.
            raise Exception(f'Model {model_id} not found at epoch {self.save_epoch}.')

        return model

    def compute_instance_scores(self, hardness_estimates: Dict[str, List[List[float]]], dataloader: DataLoader):
        """
        Compute instance-level hardness scores using the final trained model.

        Each maps to a list of per-sample scores, ordered according to dataset indices.
        """
        for new_hardness_estimate in ['iConfidence', 'iAUM', 'iDataIQ', 'iLoss', 'EL2N', 'probs']:
            hardness_estimates[new_hardness_estimate] = [[0.0 for _ in range(self.num_training_samples)]
                                                         for _ in range(self.num_models)]
        for model_id in tqdm(range(self.num_models), desc='Iterating through models.'):
            model = self.load_model(model_id)
            probe_model = self.load_model(model_id, probe=True)
            model.eval()
            probe_model.eval()
            features = []
            with torch.no_grad():
                for inputs, labels, indices in dataloader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs, latent_inputs = model(inputs, True)
                    probe_outputs, _ = probe_model(inputs, True)
                    features.append(latent_inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    for x, label, i, logits, probe_logits in zip(inputs, labels, indices, outputs, probe_outputs):
                        i = i.item()
                        correct_label = label.item()
                        logits = logits.detach()
                        probe_logits = probe_logits.detach()
                        correct_logit = logits[correct_label].item()
                        probs = torch.nn.functional.softmax(logits, dim=0)
                        probe_probs = torch.nn.functional.softmax(probe_logits, dim=0)

                        hardness_estimates['probs'][model_id][i] = probs[correct_label].item()
                        # iConfidence
                        hardness_estimates['iConfidence'][model_id][i] = correct_logit
                        # iAUM
                        max_other_logit = torch.max(torch.cat((logits[:correct_label],
                                                               logits[correct_label + 1:]))).item()
                        hardness_estimates['iAUM'][model_id][i] = correct_logit - max_other_logit
                        # iDataIQ
                        p_y = probs[correct_label].item()
                        hardness_estimates['iDataIQ'][model_id][i] = p_y * (1 - p_y)
                        # iLoss
                        label_tensor = torch.tensor([correct_label], device=logits.device)
                        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), label_tensor).item()
                        hardness_estimates['iLoss'][model_id][i] = loss
                        # EL2N
                        one_hot = F.one_hot(label, num_classes=self.num_classes).float()
                        el2n = torch.norm(probe_probs - one_hot).item()
                        hardness_estimates['EL2N'][model_id][i] = el2n

    def plot_instance_level_hardness_distributions(self, hardness_estimates):
        values = np.array(hardness_estimates['probs'])
        values = np.mean(values, axis=0)
        sorted_vals = np.sort(values)
        plt.figure(figsize=(8, 5))
        plt.plot(sorted_vals)
        plt.title(f'Sorted Hardness Scores - probs')
        plt.xlabel('Sorted Sample Index')
        plt.ylabel('Hardness Score')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_save_dir, f"sorted_hardness_probs.pdf"))
        plt.close()
        del hardness_estimates['probs']

    def get_pruned_indices(self, hardness_estimates: Dict[str, List[List[float]]]) -> \
            Dict[str, Dict[str, List[List[List[int]]]]]:
        """Extract the indices of data samples that would have been pruned if a threshold (from self.pruning_thresholds)
        was applied to different hardness estimates computed using ensembles of various sizes. This allows us to measure
        the reliability of those hardness estimates (e.g., how many models in the ensemble they need to output
        consistent pruning indices).
        """
        results = {'easy': {}, 'hard': {}}

        for metric_name, metric_scores in tqdm(hardness_estimates.items(), desc='Computing pruned indices.'):
            metric_scores = np.array(metric_scores)
            results['easy'][metric_name], results['hard'][metric_name] = [], []

            for thresh in self.pruning_thresholds:
                prune_count = int((thresh / 100) * self.num_training_samples)
                metric_easy_results, metric_hard_results = [], []

                for num_ensemble_models in range(1, self.num_models + 1):
                    # Compute the average hardness score of each sample as a function of the ensemble size
                    avg_hardness_scores = np.mean(metric_scores[:num_ensemble_models], axis=0)
                    # For AUM and Confidence, hard samples have lower values (opposite for other hardness estimators).
                    if metric_name in ['AUM', 'Confidence', 'iAUM', 'iConfidence']:
                        sorted_indices = np.argsort(-avg_hardness_scores)
                    else:
                        sorted_indices = np.argsort(avg_hardness_scores)
                    pruned_easy_indices = sorted_indices[:prune_count]
                    pruned_hard_indices = sorted_indices[-prune_count:]
                    metric_easy_results.append(pruned_easy_indices.tolist())
                    metric_hard_results.append(pruned_hard_indices.tolist())
                results['easy'][metric_name].append(metric_easy_results)
                results['hard'][metric_name].append(metric_hard_results)

        return results

    @staticmethod
    def compute_stability_of_pruning(results):
        metric_names = list(results['easy'].keys())
        num_pruning_thresholds = len(results['easy']['Confidence'])
        num_models = len(results['easy']['Confidence'][0])
        stability_results = {
            hardness_type: {
                metric_name: np.zeros((num_pruning_thresholds, num_models - 1)) for metric_name in metric_names
            } for hardness_type in ['easy', 'hard']
        }
        for metric_name in metric_names:
            for hardness_type in ['hard', 'easy']:
                metric_results = results[hardness_type][metric_name]
                for i in range(num_pruning_thresholds):
                    for j in range(num_models - 1):
                        set1 = set(metric_results[i][j])  # Pruned indices for ensemble with j models
                        set2 = set(metric_results[i][j + 1])  # Pruned indices for ensemble with j + 1 models
                        changed = len(set2 - set1) / len(set1)
                        stability_results[hardness_type][metric_name][i][j] = changed * 100
        return stability_results, num_models, num_pruning_thresholds

    def visualize_stability_of_pruning(self, stability_results, num_models, num_pruning_thresholds):
        metric_names = list(stability_results['easy'].keys())
        vmin, vmax = 0, 100  # Ensure all heatmaps share the same scale

        for metric_name in metric_names:
            for hardness_type in ['hard', 'easy']:
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
                sns.heatmap(stability_results[hardness_type][metric_name], annot=True, fmt='.2f', cmap='coolwarm',
                            cbar_kws={'label': 'Jaccard Overlap'}, vmin=vmin, vmax=vmax)  # Set color scale

                # Adjust the annotation format
                for text in plt.gca().texts:
                    text.set_text(custom_format(float(text.get_text())))

                plt.title(f'{hardness_type.capitalize()} PCR Based on {metric_name.upper()}')
                plt.xlabel('Ensemble size during hardness estimation')
                plt.ylabel('Pruning threshold (%)')
                plt.xticks(np.arange(num_models - 1) + 0.5, np.arange(1, num_models))
                plt.yticks(np.arange(num_pruning_thresholds) + 0.5, self.pruning_thresholds)
                plt.savefig(os.path.join(self.figures_save_dir,
                                         f'pruning_{hardness_type}_stability_based_on_{metric_name}.pdf'))
                plt.close()

    def plot_stability_summary_across_metrics(self, stability_results, num_models, num_pruning_thresholds):
        metric_groups = [
            ('DataIQ', 'Loss', 'AUM', 'Confidence', 'Forgetting'),
            ('iDataIQ', 'iLoss', 'iAUM', 'iConfidence', 'EL2N')
        ]
        thresholds = [0, num_pruning_thresholds - 1]  # First and last pruning threshold
        threshold_labels = ['10%', '50%']
        colors = get_cmap("tab10")  # Enough distinct colors

        for group_id, group in enumerate(metric_groups):
            for hardness_type in ['hard', 'easy']:
                plt.figure(figsize=(10, 6))
                for idx, metric_name in enumerate(group):
                    for tidx, threshold in enumerate(thresholds):
                        values = stability_results[hardness_type][metric_name][threshold]  # shape: [num_models - 1]
                        label = f"{metric_name} ({threshold_labels[tidx]})"
                        plt.plot(np.arange(1, num_models), values, label=label, color=colors(idx),
                                 linestyle='-' if tidx == 0 else '--', marker='o' if tidx == 0 else '^')

                plt.xlabel('Ensemble size during hardness estimation')
                plt.ylabel('PCR (%)')
                plt.title(f'{hardness_type.capitalize()} PCR (Group {group_id + 1})')
                plt.xticks(np.arange(1, num_models))
                plt.grid(alpha=0.3)
                plt.legend(title='Metric (Threshold)', ncol=2)
                save_name = f"stability_lineplot_{hardness_type}_group{group_id + 1}.pdf"
                plt.savefig(os.path.join(self.figures_save_dir, save_name))
                plt.close()

    def compute_overlap_of_pruned_indices_across_hardness_estimators(self, results):
        def plot_overlap(metric_pairs, hardness_type, filename_suffix):
            plt.figure(figsize=(10, 6))
            for metric1, metric2 in metric_pairs:
                overlaps = []
                for thresh_idx, thresh in enumerate(self.pruning_thresholds):
                    set1 = set(results[hardness_type][metric1][thresh_idx][-1])  # Using full ensemble
                    set2 = set(results[hardness_type][metric2][thresh_idx][-1])
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    overlap = intersection / union if union > 0 else 0.0
                    overlaps.append(overlap)

                plt.plot(self.pruning_thresholds, overlaps, label=f"{metric1} vs {metric2}", marker='o')

            plt.xlabel(f"Pruning {hardness_type} rate (% of samples removed)")
            plt.ylabel("Overlap percentage")
            plt.title(f"{self.dataset_name} ({hardness_type})")
            plt.legend(title="Metric pairs", bbox_to_anchor=(1.01, 1), loc='upper left')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_save_dir, f'overlap_{hardness_type}_{filename_suffix}.pdf'))
            plt.close()

        group1 = ('DataIQ', 'Loss', 'AUM', 'Confidence', 'Forgetting')
        group2 = ('iDataIQ', 'iLoss', 'iAUM', 'iConfidence', 'EL2N')
        direct_pairs = list(zip(('DataIQ', 'Loss', 'AUM', 'Confidence'), ('iDataIQ', 'iLoss', 'iAUM', 'iConfidence')))

        group1_pairs = list(combinations(group1, 2))
        group2_pairs = list(combinations(group2, 2))

        for hardness in ['hard', 'easy']:
            plot_overlap(group1_pairs, hardness, 'group1')
            plot_overlap(group2_pairs, hardness, 'group2')
            plot_overlap(direct_pairs, hardness, 'paired')

    def compute_effect_of_ensemble_size_on_resampling(self, hardness_estimates, dataset):
        results = {}
        for metric_name, metric_scores in tqdm(hardness_estimates.items(),
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
                    if metric_name in ['Confidence', 'AUM', 'iConfidence', 'iAUM']:
                        curr_class_hardness[label] = 1 / np.mean(curr_class_results[label])
                        next_class_hardness[label] = 1 / np.mean(next_class_results[label])
                    else:
                        curr_class_hardness[label] = np.mean(curr_class_results[label])
                        next_class_hardness[label] = np.mean(next_class_results[label])

                curr_ratios = {class_id: curr_hardness / sum(curr_class_hardness.values())
                               for class_id, curr_hardness in curr_class_hardness.items()}
                next_ratios = {class_id: next_hardness / sum(next_class_hardness.values())
                               for class_id, next_hardness in next_class_hardness.items()}

                curr_samples_per_class = {class_id: int(round(curr_ratio * self.num_training_samples))
                                          for class_id, curr_ratio in curr_ratios.items()}
                next_samples_per_class = {class_id: int(round(next_ratio * self.num_training_samples))
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
            plt.close()

            max_rel_diffs, min_rel_diffs, avg_rel_diffs = compute_stats(relative_differences[metric])
            plt.figure(figsize=(10, 6))
            plt.plot(ensemble_sizes[metric], max_rel_diffs, label=f'Max Rel Diff', linestyle='-', marker='o')
            plt.plot(ensemble_sizes[metric], min_rel_diffs, label=f'Min Rel Diff', linestyle='--', marker='x')
            plt.plot(ensemble_sizes[metric], avg_rel_diffs, label=f'Avg Rel Diff', linestyle=':', marker='s')
            plt.title(f"Based on {metric.upper()}")
            plt.xlabel('Number of Models (j) in Ensemble During Hardness Estimation')
            plt.xticks(range(1, 20, 2))
            plt.ylabel("Relative Difference in Class-Wise Sample Count for Resampling after Adding a Model")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(self.figures_save_dir, f'relative_differences_{metric}.pdf'))
            plt.close()

    def save_hardness_estimates(self, hardness_estimates):
        hardness_save_dir = os.path.join(ROOT, f"Results/{self.data_cleanliness}{self.dataset_name}/")
        path = os.path.join(hardness_save_dir, 'hardness_estimates.pkl')
        with open(path, "wb") as file:
            print(f'Saving updated hardness estimates.')
            pickle.dump(hardness_estimates, file)

    def main(self):
        config = get_config(self.dataset_name)
        _, training_dataset, _, _ = load_dataset(args.dataset_name, self.data_cleanliness == 'clean', False, False)
        new_training_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(config['mean'], config['std']),
        ])
        training_dataset = AugmentedSubset(training_dataset, transform=new_training_transform)
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1000, shuffle=False)
        hardness_estimates = load_hardness_estimates(self.data_cleanliness, self.dataset_name)
        if 'iConfidence' not in hardness_estimates.keys():
            self.compute_instance_scores(hardness_estimates, training_loader)
        self.plot_instance_level_hardness_distributions(hardness_estimates)

        print('All of the below should have the same dimensions. Otherwise, there is something wrong with the code.')
        for key in hardness_estimates.keys():
            print(f'Shape of hardness estimated via {key}: {len(hardness_estimates[key])}, '
                  f'{len(hardness_estimates[key][0])}')
        print()

        # pruned_indices[hardness_type][metric_name][pruning_threshold][num_ensemble_models][pruned_indices]
        pruned_indices = self.get_pruned_indices(hardness_estimates)
        stability_results, num_models, num_pruning_thresholds = self.compute_stability_of_pruning(pruned_indices)
        self.visualize_stability_of_pruning(stability_results, num_models, num_pruning_thresholds)
        self.plot_stability_summary_across_metrics(stability_results, num_models, num_pruning_thresholds)
        self.compute_overlap_of_pruned_indices_across_hardness_estimators(pruned_indices)

        differences = self.compute_effect_of_ensemble_size_on_resampling(hardness_estimates, training_loader.dataset)
        self.visualize_stability_of_resampling(differences)
        self.save_hardness_estimates(hardness_estimates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Hardness of Dataset Samples')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='Specify the dataset name (default: CIFAR10)')
    parser.add_argument('--remove_noise', action='store_true', help='Raise this flag to remove noise from the data.')
    args = parser.parse_args()

    Visualizer(args.dataset_name, args.remove_noise).main()

"""
1. (+) Change the visualization from heatmaps to plots (maybe do both).
2. (+) Find a way to make the overlap_of_pruned_indices figure clear.
3. (+) Modify EL2N to use the probe network
"""