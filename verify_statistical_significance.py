"""This is the first visualization module that focuses on the results of experiment1.py.

Main purpose:
* Compute instant metrics (iConfidence, iAUM, iLoss, iDataIQ, and EL2N)
* Measure stability and robustness of hardness estimates to the changes in ensemble size.

Important information:
* Make sure that `num_models_per_dataset` from config.py is set to the same number as when running experiment1.py!
* The `num_datasets` from config.py can be set to a different value than it was during experiment1.py (in fact it's
advised). Back then we set it to one as there is only one original version of the dataset. Here, this variable is used
to determine the number of subensembles that will be used to compute the means for various visualizations. That is
because we do not use all the trained models. We take a collection of random subensembles for robustness.
"""

import argparse
from itertools import combinations
import os
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from config import DEVICE, get_config, ROOT
from data import AugmentedSubset, load_dataset
from neural_networks import ResNet18LowRes
from utils import compute_sample_allocation_after_resampling, load_hardness_estimates, restructure_hardness_dictionary


class Visualizer:
    """Encapsulates all the necessary methods to perform the visualization and computation of `instant` hardness
    estimates."""
    def __init__(self, dataset_name: str, remove_noise: bool):
        """Initialize the Visualizer class responsible for visualizing the robustness of hardness estimates produced by
        experiment1.py in regard to the ensemble size.

        :param dataset_name: Name of the dataset
        :param remove_noise: If experiment1.py was run with this parameter raised than it also has to be raised here.
        Otherwise, keep it as it.
        """
        self.dataset_name = dataset_name
        self.data_cleanliness = 'clean' if remove_noise else 'unclean'

        config = get_config(dataset_name)
        self.num_classes = config['num_classes']
        self.num_epochs = config['num_epochs']
        self.num_training_samples = sum(config['num_training_samples'])
        self.model_dir = config['save_dir']
        self.save_epoch = config['save_epoch']
        self.num_models_per_dataset = config['num_models_per_dataset']
        self.num_datasets = config['num_datasets']

        self.results_save_dir = os.path.join(ROOT, 'Results/', f'{self.data_cleanliness}{dataset_name}')
        self.figures_save_dir = os.path.join(ROOT, 'Figures/', f'{self.data_cleanliness}{dataset_name}')
        for save_dir in [self.results_save_dir, self.figures_save_dir]:
            os.makedirs(save_dir, exist_ok=True)

        self.pruning_thresholds = np.array([10, 20, 30, 40, 50])

    def load_model(self, model_id: int, probe=False) -> ResNet18LowRes:
        """Used to load pretrained models."""
        model = ResNet18LowRes(num_classes=self.num_classes).to(DEVICE)
        epoch = self.save_epoch if probe else self.num_epochs
        model_path = os.path.join(self.model_dir, 'none', f"{self.data_cleanliness}{args.dataset_name}",
                                  f'dataset_0_model_{model_id}_epoch_{epoch}.pth')

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            # This code can only be run if models were pretrained. If no pretrained models are found, throw error.
            raise Exception(f'Model {model_id} not found at epoch {self.save_epoch}.')

        return model

    def compute_instance_scores(self, hardness_estimates: Dict[Tuple[int, int], Dict[str, List[float]]],
                                dataloader: DataLoader):
        """
        Compute instance-level hardness scores using the final trained model.

        Each maps to a list of per-sample scores, ordered according to dataset indices.
        """
        for model_id in tqdm(range(self.num_models_per_dataset), desc='Iterating through models.'):
            for new_hardness_estimate in ['iConfidence', 'iAUM', 'iDataIQ', 'iLoss', 'EL2N', 'probs']:
                hardness_estimates[(0, model_id)][new_hardness_estimate] = []
            model = self.load_model(model_id)
            probe_model = self.load_model(model_id, probe=True)
            model.eval()
            probe_model.eval()
            with torch.no_grad():
                for inputs, labels, _ in dataloader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    probe_outputs = probe_model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    for x, y, logits, probe_logits in zip(inputs, labels, outputs, probe_outputs):
                        y_hat = y.item()
                        logits = logits.detach()
                        probe_logits = probe_logits.detach()
                        correct_logit = logits[y_hat].item()
                        probs = torch.nn.functional.softmax(logits, dim=0)
                        probe_probs = torch.nn.functional.softmax(probe_logits, dim=0)

                        hardness_estimates[(0, model_id)]['probs'].append(probs[y_hat].item())
                        # iConfidence
                        hardness_estimates[(0, model_id)]['iConfidence'].append(correct_logit)
                        # iAUM
                        max_other_logit = torch.max(torch.cat((logits[:y_hat], logits[y_hat + 1:]))).item()
                        aum = correct_logit - max_other_logit
                        hardness_estimates[(0, model_id)]['iAUM'].append(aum)
                        # iDataIQ
                        p_y = probs[y_hat].item()
                        uncertainty = p_y * (1 - p_y)
                        hardness_estimates[(0, model_id)]['iDataIQ'].append(uncertainty)
                        # iLoss
                        label_tensor = torch.tensor([y_hat], device=logits.device)
                        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), label_tensor).item()
                        hardness_estimates[(0, model_id)]['iLoss'].append(loss)
                        # EL2N
                        one_hot = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()
                        el2n = torch.norm(probe_probs - one_hot).item()
                        hardness_estimates[(0, model_id)]['EL2N'].append(el2n)

    def save_hardness_estimates(self, hardness_estimates: Dict[Tuple[int, int], Dict[str, List[float]]]):
        """Save the updated hardness estimates."""
        hardness_save_dir = os.path.join(ROOT, f"Results/{self.data_cleanliness}{self.dataset_name}/")
        path = os.path.join(hardness_save_dir, 'hardness_estimates.pkl')
        with open(path, "wb") as file:
            print(f'Saving updated hardness estimates.')
            # noinspection PyTypeChecker
            pickle.dump(hardness_estimates, file)

    def plot_instance_level_hardness_distributions(self,
                                                   hardness_estimates: Dict[Tuple[int, int], Dict[str, List[float]]]):
        """The purpose of this visualization is to see the distribution of `probs`. This is required to validate the
        equivalence of iLoss and iDataIQ results."""
        values = [hardness_estimates[(0, model_id)]['probs'] for model_id in range(len(hardness_estimates))]
        values = np.mean(np.array(values), axis=0)
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

    def get_pruned_indices(self, hardness_estimates: Dict[Tuple[int, int], Dict[str, List[float]]]
                           ) -> Dict[str, Dict[str, List[List[List[List[int]]]]]]:
        """Extract the indices of data samples that would have been pruned if a threshold (from self.pruning_thresholds)
        was applied to different hardness estimates computed using ensembles of various sizes. This allows us to measure
        the reliability of those hardness estimates (e.g., how many models in the ensemble they need to output
        consistent pruning indices). We repeat the results for `self.num_datasets` random subensemble to add robustness
        to our findings.
        """
        pruned_indices = {'easy': {}, 'hard': {}}
        restructured_hardness_estimates = restructure_hardness_dictionary(hardness_estimates)
        for metric_name, metric_scores in tqdm(restructured_hardness_estimates.items(),
                                               desc='Computing pruned indices.'):
            if metric_name == 'probs':
                continue
            metric_scores = np.array(metric_scores)
            pruned_indices['easy'][metric_name], pruned_indices['hard'][metric_name] = [], []
            for thresh_i, (thresh) in enumerate(self.pruning_thresholds):
                prune_count = int((thresh / 100) * self.num_training_samples)
                pruned_indices['easy'][metric_name].append([])
                pruned_indices['hard'][metric_name].append([])
                max_ensemble_size = len(metric_scores)
                # Iterate through different subensemble sizes
                for subensemble_size in range(1, max_ensemble_size // 2 + 1):
                    pruned_indices['easy'][metric_name][thresh_i].append([])
                    pruned_indices['hard'][metric_name][thresh_i].append([])
                    for _ in range(self.num_datasets):
                        # Produce random subensemble
                        subensemble_indices = np.random.choice(range(max_ensemble_size), subensemble_size,
                                                               replace=False)
                        subensemble_scores = metric_scores[subensemble_indices]
                        # Compute the average hardness score of each sample as a function of the ensemble size
                        avg_hardness_scores = np.mean(subensemble_scores, axis=0)
                        # For AUM & Confidence, hard samples have lower values (opposite for other hardness estimators).
                        if metric_name in ['AUM', 'Confidence', 'iAUM', 'iConfidence']:
                            sorted_indices = np.argsort(-avg_hardness_scores)
                        else:
                            sorted_indices = np.argsort(avg_hardness_scores)
                        pruned_easy_indices = sorted_indices[:prune_count]
                        pruned_hard_indices = sorted_indices[-prune_count:]
                        pruned_indices['easy'][metric_name][thresh_i][subensemble_size - 1].append(
                            pruned_easy_indices.tolist())
                        pruned_indices['hard'][metric_name][thresh_i][subensemble_size - 1].append(
                            pruned_hard_indices.tolist())
        return pruned_indices

    def compute_stability_of_pruning(self, pruned_indices: Dict[str, Dict[str, List[List[List[List[int]]]]]],
                                     num_subensembles: int, num_pruning_thresholds: int
                                     ) -> Dict[str, Dict[str, List[List[List[float]]]]]:
        """Computes the stability of the pruning indices by measuring the percentage change between two sets of pruned
        indices, where one was obtained using j models and another using j+1 models."""
        metric_names = list(pruned_indices['easy'].keys())

        stability_results = {
            hardness_type: {
                metric_name: [[[] for _ in range(num_subensembles - 1)] for _ in range(num_pruning_thresholds)]
                for metric_name in metric_names
            } for hardness_type in ['easy', 'hard']
        }

        for metric_name in metric_names:
            for hardness_type in ['hard', 'easy']:
                for i in range(num_pruning_thresholds):
                    for j in range(num_subensembles - 1):
                        for subensemble_idx in range(self.num_datasets):
                            set1 = set(pruned_indices[hardness_type][metric_name][i][j][subensemble_idx])
                            set2 = set(pruned_indices[hardness_type][metric_name][i][j + 1][subensemble_idx])
                            changed = len(set2 - set1) / len(set1)
                            stability_results[hardness_type][metric_name][i][j].append(changed * 100)
        return stability_results

    def visualize_stability_of_pruning_via_heatmap(self, num_subensembles: int, num_pruning_thresholds: int,
                                                   stability_results: Dict[str, Dict[str, List[List[List[float]]]]]):
        """This visualization is a heatmap showing the stability of pruned indices as a function of ensemble size and
        pruning rate. We used it in the TPAMI paper. This only portrays the average stability (hard to put std here)."""

        def custom_format(val):
            """Helper function that modifies the format in which values are reported for visual clarity."""
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

        metric_names = list(stability_results['easy'].keys())
        v_min, v_max = 0, 100  # Ensure all heatmaps share the same scale

        for metric_name in metric_names:
            for hardness_type in ['hard', 'easy']:
                # Convert 3D list (threshold × ensemble_size × subensembles) into a 2D array of averages
                avg_matrix = np.zeros((num_pruning_thresholds, num_subensembles - 1))
                for i in range(num_pruning_thresholds):
                    for j in range(num_subensembles - 1):
                        avg_matrix[i, j] = np.mean(stability_results[hardness_type][metric_name][i][j])

                # Create figure and plot the heatmap
                plt.figure(figsize=(10, 6))
                sns.heatmap(avg_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                            cbar_kws={'label': 'Jaccard Overlap'}, vmin=v_min, vmax=v_max)  # Set color scale

                # Adjust the annotation format
                for text in plt.gca().texts:
                    text.set_text(custom_format(float(text.get_text())))

                plt.title(f'{hardness_type.capitalize()} PCR Based on {metric_name.upper()}')
                plt.xlabel('Ensemble size during hardness estimation')
                plt.ylabel('Pruning threshold (%)')
                plt.xticks(np.arange(num_subensembles - 1) + 0.5, np.arange(1, num_subensembles))
                plt.yticks(np.arange(num_pruning_thresholds) + 0.5, self.pruning_thresholds)
                plt.savefig(os.path.join(self.figures_save_dir,
                                         f'pruning_{hardness_type}_stability_based_on_{metric_name}.pdf'))
                plt.close()

    def visualize_stability_of_pruning_via_plots(self, num_subensembles: int, num_pruning_thresholds: int,
                                                 stability_results: Dict[str, Dict[str, List[List[List[float]]]]]):
        """This visualization shows the Pruning Change Rate (PCR) across ensemble sizes. High PCR indicate that adding a
        model to the ensemble will significantly change the pruned indices. This is a more visually appealing version of
        the heatmap from visualize_stability_of_pruning, however it conveys information for only selected pruning rates.
        """

        metric_groups = [
            ('DataIQ', 'Loss', 'AUM', 'Confidence', 'Forgetting'),
            ('iDataIQ', 'iLoss', 'iAUM', 'iConfidence', 'EL2N')
        ]
        thresholds = [0, num_pruning_thresholds - 1]  # First and last pruning threshold
        threshold_labels = ['10%', '50%']
        colors = matplotlib.colormaps["tab10"]

        for group_id, group in enumerate(metric_groups):
            for hardness_type in ['hard', 'easy']:
                plt.figure(figsize=(10, 6))
                for idx, metric_name in enumerate(group):
                    for t_idx, threshold in enumerate(thresholds):
                        avg_values = np.mean(np.array(stability_results[hardness_type][metric_name][threshold]), axis=1)
                        std = np.std(np.array(stability_results[hardness_type][metric_name][threshold]), axis=1)
                        label = f"{metric_name} ({threshold_labels[t_idx]})"
                        plt.plot(np.arange(1, num_subensembles), avg_values, label=label, color=colors(idx),
                                 linestyle='-' if t_idx == 0 else '--', marker='o' if t_idx == 0 else '^')
                        plt.fill_between(np.arange(1, num_subensembles), avg_values - std, avg_values + std,
                                         color=colors(idx), alpha=0.15, linewidth=0)

                plt.xlabel('Ensemble size M passed to PCR')
                plt.ylabel('Pruning Change Rate (PCR)')
                plt.title(f'{hardness_type.capitalize()} PCR (Group {group_id + 1})')
                plt.xticks(np.arange(1, num_subensembles))
                plt.grid(alpha=0.3)
                plt.legend(title='Metric (Threshold)', ncol=2)
                save_name = f"stability_lineplot_{hardness_type}_group{group_id + 1}.pdf"
                plt.savefig(os.path.join(self.figures_save_dir, save_name))
                plt.close()

    def visualize_overlap_between_pruned_indices(self, num_pruning_thresholds: int,
                                                 pruned_indices: Dict[str, Dict[str, List[List[List[List[int]]]]]]):
        """This visualization shows the overlap between the pruned indices in the hope of estimating the harm caused by
        inappropriate choice of hardness estimator. Higher overlap indicates that changing hardness estimator has
        negligible impact on pruning."""
        def plot_overlap(metric_pairs: List[Tuple[str, str]], hardness_type: str, filename_suffix: str):
            """Helper function for procuring this particular plot"""
            colors = matplotlib.colormaps["tab10"]
            plt.figure(figsize=(10, 6))
            for idx, (metric1, metric2) in enumerate(metric_pairs):
                overlaps, overlaps_std = [], []
                for thresh_idx in range(num_pruning_thresholds):
                    subensemble_overlaps = []
                    for subensemble_idx in range(self.num_datasets):
                        # Compute the overlap only for the largest subensembles (-1 part).
                        set1 = set(pruned_indices[hardness_type][metric1][thresh_idx][-1][subensemble_idx])
                        set2 = set(pruned_indices[hardness_type][metric2][thresh_idx][-1][subensemble_idx])
                        intersection = len(set1 & set2)
                        union = len(set1 | set2)
                        overlap = intersection / union if union > 0 else 0.0
                        subensemble_overlaps.append(overlap)
                    overlaps.append(np.mean(subensemble_overlaps))
                    overlaps_std.append(np.std(subensemble_overlaps))

                overlaps, overlaps_std = np.array(overlaps), np.array(overlaps_std)
                plt.plot(self.pruning_thresholds, overlaps, label=f"{metric1} vs {metric2}", marker='o',
                         color=colors(idx))
                plt.fill_between(self.pruning_thresholds, overlaps - overlaps_std, overlaps + overlaps_std,
                                 color=colors(idx), alpha=0.15, linewidth=0)

            plt.xlabel(f"Pruning threshold percentages")
            plt.ylabel("Jaccard Overlap")
            plt.title(f"{self.dataset_name} ({hardness_type})")
            plt.legend(title="Metric pairs", bbox_to_anchor=(1.01, 1), loc='upper left')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_save_dir, f'overlap_{hardness_type}_{filename_suffix}.pdf'))
            plt.close()

        group1 = ['DataIQ', 'Loss', 'AUM', 'Confidence', 'Forgetting']
        group2 = ['iDataIQ', 'iLoss', 'iAUM', 'iConfidence', 'EL2N']
        direct_pairs = list(zip(['DataIQ', 'Loss', 'AUM', 'Confidence'], ['iDataIQ', 'iLoss', 'iAUM', 'iConfidence']))

        group1_pairs = list(combinations(group1, 2))
        group2_pairs = list(combinations(group2, 2))

        for hardness in ['hard', 'easy']:
            plot_overlap(group1_pairs, hardness, 'group1')
            plot_overlap(group2_pairs, hardness, 'group2')
            plot_overlap(direct_pairs, hardness, 'paired')

    def compute_effect_of_ensemble_size_on_resampling(self, labels: List[int],
                                                      hardness_estimates: Dict[Tuple[int, int], Dict[str, List[float]]]
                                                      ) -> Dict[str, List[List[float]]]:
        """This computes the changes to the sample count in each class after resampling when comparing resampling
        performed using hardness estimated obtained from j and j+1 models. The format of the output is
        abs_diffs[estimator_name][subensemble_size][0] = absolute difference averaged over classes
        """
        abs_diffs = {}
        for estimator_name in tqdm(hardness_estimates[(0, 0)].keys(), desc='Computing effect of ensemble size on '
                                                                           'resampling.'):
            hardness_over_models = [hardness_estimates[(0, model_id)][estimator_name]
                                    for model_id in range(len(hardness_estimates))]
            max_ensemble_size = len(hardness_over_models)
            abs_diffs[estimator_name] = []
            # We don't add 1 here to have the same size of the X-axis as in the previous visualizations.
            for subensemble_size in range(1, max_ensemble_size // 2):
                abs_diffs[estimator_name].append([])
                for _ in range(self.num_datasets):
                    curr_subensemble_indices = np.random.choice(range(max_ensemble_size), subensemble_size,
                                                                replace=False)
                    next_subensemble_indices = np.random.choice(range(max_ensemble_size), subensemble_size + 1,
                                                                replace=False)
                    curr_subensemble_estimates = np.array(hardness_over_models)[curr_subensemble_indices]
                    next_subensemble_estimates = np.array(hardness_over_models)[next_subensemble_indices]
                    curr_avg_hardness_scores = list(np.mean(curr_subensemble_estimates, axis=0))
                    next_avg_hardness_scores = list(np.mean(next_subensemble_estimates, axis=0))
                    curr_samples_per_class, _ = compute_sample_allocation_after_resampling(curr_avg_hardness_scores,
                                                                                           labels,
                                                                                           self.num_classes,
                                                                                           self.num_training_samples,
                                                                                           estimator_name)
                    next_samples_per_class, _ = compute_sample_allocation_after_resampling(next_avg_hardness_scores,
                                                                                           labels,
                                                                                           self.num_classes,
                                                                                           self.num_training_samples,
                                                                                           estimator_name)
                    differences = np.mean([abs(next_samples_per_class[k] - curr_samples_per_class[k])
                                           for k in range(self.num_classes)])
                    abs_diffs[estimator_name][subensemble_size - 1].append(differences)
        return abs_diffs

    def visualize_stability_of_resampling(self, abs_diffs: Dict[str, List[List[float]]], num_subensembles: int):
        """This visualization measures the checks the stability of hardness estimates by focusing on the changes to the
        resampling ratios. In other words, what is the average change in per-class sample count after resampling if we
        had access to one more model for hardness estimation."""

        group1 = ('DataIQ', 'Loss', 'AUM', 'Confidence', 'Forgetting')
        group2 = ('iDataIQ', 'iLoss', 'iAUM', 'iConfidence', 'EL2N')
        colors = matplotlib.colormaps["tab10"]

        for i, group in enumerate([group1, group2]):
            plt.figure(figsize=(10, 6))
            for idx, estimator_name in enumerate(group):
                avg_diffs = np.mean(np.array(abs_diffs[estimator_name]), axis=1)
                std_diffs = np.std(np.array(abs_diffs[estimator_name]), axis=1)
                plt.plot(np.arange(1, num_subensembles), avg_diffs, label=f'{estimator_name}', marker='s',
                         color=colors(idx))
                plt.fill_between(np.arange(1, num_subensembles), avg_diffs - std_diffs, avg_diffs + std_diffs,
                                 color=colors(idx), alpha=0.2, linewidth=0)
            plt.xlabel('Ensemble size M passed to Absolute Difference')
            plt.xticks(np.arange(1, num_subensembles))
            plt.ylabel("Average Absolute Difference across all classes")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(self.figures_save_dir, f'absolute_differences_group{i+1}.pdf'))
            plt.close()

    def main(self):
        """Main method for producing the visualizations."""
        config = get_config(self.dataset_name)
        _, training_dataset, _, _ = load_dataset(args.dataset_name, self.data_cleanliness == 'clean')

        # We want to work with normalized but unaugmented images in this module.
        new_training_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(config['mean'], config['std']),
        ])
        training_dataset = AugmentedSubset(training_dataset, transform=new_training_transform)
        labels = [training_dataset[idx][1].item() for idx in range(len(training_dataset))]
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1000, shuffle=False)

        hardness_estimates = load_hardness_estimates(self.data_cleanliness, self.dataset_name)
        if 'probs' not in hardness_estimates[(0, 0)].keys():
            self.compute_instance_scores(hardness_estimates, training_loader)
            self.save_hardness_estimates(hardness_estimates)
        self.plot_instance_level_hardness_distributions(hardness_estimates)

        # pruned_indices[hardness_type][metric_name][pruning_threshold][subensemble_size][subensemble][pruned_indices]
        pruned_indices = self.get_pruned_indices(hardness_estimates)
        num_pruning_thresholds = len(pruned_indices['easy']['Confidence'])
        num_subensembles = len(pruned_indices['easy']['Confidence'][0])
        print(f'Continuing with {num_subensembles} subensemble sizes.')

        stability_results = self.compute_stability_of_pruning(pruned_indices, num_subensembles, num_pruning_thresholds)
        self.visualize_stability_of_pruning_via_heatmap(num_subensembles, num_pruning_thresholds, stability_results)
        self.visualize_stability_of_pruning_via_plots(num_subensembles, num_pruning_thresholds, stability_results)
        self.visualize_overlap_between_pruned_indices(num_pruning_thresholds, pruned_indices)

        absolute_differences = self.compute_effect_of_ensemble_size_on_resampling(labels, hardness_estimates)
        self.visualize_stability_of_resampling(absolute_differences, num_subensembles)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Hardness of Dataset Samples')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='Specify the dataset name (default: CIFAR10)')
    parser.add_argument('--remove_noise', action='store_true', help='Raise this flag to remove noise from the data.')
    args = parser.parse_args()

    Visualizer(args.dataset_name, args.remove_noise).main()
