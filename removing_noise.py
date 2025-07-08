import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import ROOT
from utils import load_hardness_estimates


class NoiseRemover:
    def __init__(self, config, dataset_name, dataset):
        self.config = config
        self.dataset_name = dataset_name
        self.dataset = dataset

        self.BATCH_SIZE = self.config['batch_size']
        self.ROBUST_NUM_MODELS = self.config['robust_ensemble_size']
        self.TOTAL_SAMPLES = sum(self.config['num_training_samples'])
        self.NUM_EPOCHS = self.config['num_epochs']

        self.figure_save_dir = os.path.join(ROOT, 'Figures/', f"clean{self.dataset_name}")
        os.makedirs(self.figure_save_dir, exist_ok=True)

    def compute_and_visualize_stability_of_noise_removal(self, results):
        num_models = len(results)
        overlaps = []
        for num_ensemble_models in range(1, num_models):
            cur_average_results = np.mean(results[:num_ensemble_models], axis=0)
            next_average_results = np.mean(results[:num_ensemble_models + 1], axis=0)
            cur_sorted_data = np.sort(cur_average_results)
            next_sorted_data = np.sort(next_average_results)
            x_value = 1.5
            cur_elbow_index = np.searchsorted(cur_sorted_data, x_value, side='left')
            next_elbow_index = np.searchsorted(next_sorted_data, x_value, side='left')
            cur_sorted_indices = np.argsort(cur_average_results)
            next_sorted_indices = np.argsort(next_average_results)
            cur_removed_indices = set(cur_sorted_indices[:cur_elbow_index])
            next_removed_indices = set(next_sorted_indices[:next_elbow_index])

            intersection = len(cur_removed_indices & next_removed_indices)
            union = len(cur_removed_indices | next_removed_indices)
            overlap = intersection / union if union > 0 else 0.0
            overlaps.append(overlap)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_models), overlaps)
        plt.xlabel('Number of models in ensemble (j) before adding a model')
        plt.ylabel('Overlap between removed indices')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_save_dir, 'stability_of_noise_removal.pdf'))

    def plot_cumulative_distribution(self, data):
        sorted_data = np.sort(data)
        cumulative_sum = np.cumsum(sorted_data)
        total_sum = cumulative_sum[-1]
        cumulative_percentage = (cumulative_sum / total_sum) * 100

        # Plot the cumulative distribution
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_data, cumulative_percentage, label="Cumulative Distribution")
        plt.xlabel("Instance-level hardness estimated (AUM)")
        plt.ylabel("Cumulative distribution of AUM")
        plt.grid(alpha=0.5)

        # Grey dot: Alternative threshold at 12% cumulative percentage
        num_to_remove = int(0.12 * len(sorted_data))  # 12% of total data
        x_threshold = sorted_data[num_to_remove]  # x-value at this index
        y_threshold = cumulative_percentage[num_to_remove]  # Corresponding cumulative percentage
        plt.scatter([x_threshold], [y_threshold], color='grey', zorder=5, label=f"Pleiss et al.'s Threshold")

        # Red dot: Predefined threshold (elbow method)
        num_to_remove = 552  # TODO: This is definitely wrong - needs to be different for different datasets!!!
        x_value = sorted_data[num_to_remove]
        y_value = cumulative_percentage[num_to_remove]
        plt.scatter([x_value], [y_value], color='red', zorder=5, label=f'Our Threshold')

        plt.legend()
        plt.savefig(os.path.join(self.figure_save_dir, 'AUM_distribution.pdf'))

        return num_to_remove

    def plot_removed_samples_distribution(self, removed_indices):
        removed_labels = [self.dataset[idx][1] for idx in removed_indices]
        class_counts = np.zeros(self.config['num_classes'], dtype=int)
        for label in removed_labels:
            class_counts[label] += 1
        percentage_class_counts = class_counts / np.array(self.config['num_training_samples'])

        class_names = self.config['class_names']
        x = np.arange(self.config['num_classes'])

        plt.figure(figsize=(8, 5))
        plt.bar(x, percentage_class_counts, color='skyblue')
        plt.xticks(x, class_names, rotation=90)
        plt.xlabel("Classes")
        plt.ylabel("Samples classified as noise (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.join(self.figure_save_dir, "removed_samples_distribution_by_class.pdf")))
        plt.close()

        sorted_indices = np.argsort(percentage_class_counts)
        sorted_percentages = percentage_class_counts[sorted_indices]

        plt.figure(figsize=(8, 5))
        plt.bar(range(len(sorted_percentages)), sorted_percentages, color='skyblue')
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        plt.xlabel("Classes sorted by estimated number of label noise")
        plt.ylabel("Samples classified as noise (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_save_dir, "sorted_removed_samples_distribution_by_class.pdf"))
        plt.close()

    def visualize_lowest_aum_samples(self, indices, aum_values):
        num_samples_to_visualize = min(30, len(indices))
        fig, axes = plt.subplots(3, 10, figsize=(20, 8))
        fig.suptitle("Samples with Lowest AUM Values", fontsize=16)

        mean = np.array(self.config['mean'])
        std = np.array(self.config['std'])
        class_names = self.config['class_names']

        for i, idx in enumerate(indices[:num_samples_to_visualize]):
            data, label, _ = self.dataset[idx]
            aum = aum_values[idx]
            if isinstance(data, torch.Tensor):
                data = data.numpy()

            # We need to unnormalize the images before plotting for them to be recognizable.
            data = (data * std[:, None, None]) + mean[:, None, None]

            # This is done, because it's necessary for the channel to be the last dimension, not first.
            if data.ndim == 3 and data.shape[0] in [1, 2, 3]:
                data = np.transpose(data, (1, 2, 0))

            data = np.clip(data, 0, 1)
            cmap = "gray" if data.ndim == 2 or (data.ndim == 3 and data.shape[2] == 1) else None

            row, col = divmod(i, 10)
            axes[row, col].imshow(data.squeeze(), cmap=cmap)
            axes[row, col].set_title(f"Class: {class_names[label]}\nAUM: {aum:.2f}\nIdx: {idx}")
            axes[row, col].axis("off")

        plt.savefig(os.path.join(self.figure_save_dir, "lowest_AUM_samples.pdf"))

    def clean(self):
        aum_scores_over_models = load_hardness_estimates('unclean', self.dataset_name, 'AUM')
        self.compute_and_visualize_stability_of_noise_removal(aum_scores_over_models)

        aum_scores = np.mean(np.array(aum_scores_over_models[:self.ROBUST_NUM_MODELS]), axis=0)
        elbow_index = self.plot_cumulative_distribution(aum_scores)
        sorted_indices = np.argsort(aum_scores)

        print(f'Removing {elbow_index} data samples that were identified to be misclassified.')
        removed_indices = sorted_indices[:elbow_index]
        retained_indices = np.setdiff1d(np.arange(len(self.dataset)), removed_indices)
        self.plot_removed_samples_distribution(removed_indices)
        self.visualize_lowest_aum_samples(removed_indices, aum_scores)

        return retained_indices
