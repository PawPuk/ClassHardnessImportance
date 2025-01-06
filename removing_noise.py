import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import get_config, AugmentedSubset, IndexedDataset


class NoiseRemover:
    def __init__(self, dataset_name, dataset):
        self.dataset_name = dataset_name
        self.dataset = dataset

        self.config = get_config(dataset_name)
        self.BATCH_SIZE = self.config['batch_size']
        self.TOTAL_SAMPLES = sum(self.config['num_training_samples'])
        self.NUM_EPOCHS = self.config['num_epochs']

        self.figure_save_dir = os.path.join('Figures/', f"clean{self.dataset_name}")
        os.makedirs(self.figure_save_dir, exist_ok=True)

    @staticmethod
    def load_pickle(file_path):
        """Load data from a pickle file."""
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def plot_cumulative_distribution(self, data, title="Cumulative Distribution with Elbow"):
        """
        Plots the cumulative distribution, first derivative, second derivative and highlights the elbow point.

        Args:
            data (list or np.ndarray): The data samples to be plotted.
            title (str): Title of the plot.
        """
        sorted_data = np.sort(data)
        cumulative_percentage = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100

        # Plot the cumulative distribution
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_data, cumulative_percentage, label="Cumulative Distribution")
        plt.xlabel("Sample Value (Sorted)")
        plt.ylabel("Cumulative Percentage (%)")
        plt.title(title)
        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig('AUM_distribution.pdf')

        # This value is chosen manually, so that there should be some hyper-tuning, but I'll leave that for later.
        x_value = 1.5
        y_value = np.interp(x_value, sorted_data, cumulative_percentage)
        elbow_index = np.searchsorted(sorted_data, x_value, side='left')
        plt.axhline(y=y_value, color='red', linestyle='--', label=f'Value at x = {x_value}')
        plt.scatter([x_value], [y_value], color='red', zorder=5)  # Label the point for clarity
        plt.legend()
        plt.savefig(os.path.join(self.figure_save_dir, 'AUM_distribution.pdf'))

        return elbow_index

    def plot_removed_samples_distribution(self, removed_indices):
        """
        Plots the distribution of removed samples by class.

        Args:
            removed_indices (list or np.ndarray): Indices of the removed samples.
        """
        # Extract class labels of the removed samples
        removed_labels = [self.dataset[idx][1] for idx in removed_indices]

        # Count occurrences of each class in removed samples
        class_counts = np.zeros(self.config['num_classes'], dtype=int)
        for label in removed_labels:
            class_counts[label] += 1

        # Normalize counts if desired (e.g., relative to class size)
        if 'num_training_samples' in self.config:
            class_counts = class_counts / np.array(self.config['num_training_samples'])

        # Prepare data for plotting
        class_names = self.config['class_names']
        x = np.arange(self.config['num_classes'])

        # Plot the distribution
        plt.figure(figsize=(15, 6))
        plt.bar(x, class_counts, color='skyblue', edgecolor='black')
        plt.xticks(x, class_names, rotation=90)
        plt.xlabel("Class")
        plt.ylabel(
            "Proportion of Removed Samples" if 'num_training_samples' in self.config else "Count of Removed Samples")
        plt.title("Distribution of Removed Samples by Class")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.join(self.figure_save_dir, "removed_samples_distribution_by_class.pdf")))

    def visualize_lowest_aum_samples(self, indices, aum_values):
        """
        Visualizes the 20 samples with the lowest AUM values along with their classes and AUM values.

        Args:
            indices (list or np.ndarray): Indices of the samples with the lowest AUM values.
            aum_values (list or np.ndarray): The AUM values corresponding to the dataset samples.
        """
        num_samples_to_visualize = min(20, len(indices))
        fig, axes = plt.subplots(2, 10, figsize=(20, 5))
        fig.suptitle("Samples with Lowest AUM Values", fontsize=16)

        mean = np.array(self.config['mean'])
        std = np.array(self.config['std'])
        class_names = self.config['class_names']

        for i, idx in enumerate(indices[:num_samples_to_visualize]):
            data, label, _ = self.dataset[idx]
            aum = aum_values[idx]

            # If data is a tensor, convert it to a NumPy array
            if isinstance(data, torch.Tensor):
                data = data.numpy()

            # We need to unnormalize the images before plotting for them to be recognizable.
            data = (data * std[:, None, None]) + mean[:, None, None]

            # This is done, because it's necessary for the channel to be the last dimension, not first.
            if data.ndim == 3 and data.shape[0] in [1, 3]:
                data = np.transpose(data, (1, 2, 0))

            data = np.clip(data, 0, 1)
            cmap = "gray" if data.ndim == 2 or (data.ndim == 3 and data.shape[2] == 1) else None

            row, col = divmod(i, 10)
            axes[row, col].imshow(data.squeeze(), cmap=cmap)
            axes[row, col].set_title(f"Class: {class_names[label]}\nAUM: {aum:.2f}")
            axes[row, col].axis("off")

        plt.savefig(os.path.join(self.figure_save_dir, "lowest_AUM_samples.pdf"))

    def clean(self):
        file = f"Results/{self.dataset_name}/AUM.pkl"
        AUM_over_epochs_and_models = np.array(self.load_pickle(file))

        AUM_over_epochs = [
            [
                sum(model_list[sample_idx][epoch_idx] for model_list in AUM_over_epochs_and_models) / len(
                    AUM_over_epochs_and_models) for epoch_idx in range(self.NUM_EPOCHS)
            ]
            for sample_idx in range(self.TOTAL_SAMPLES)
        ]
        AUM = np.mean(AUM_over_epochs, axis=1)

        elbow_index = self.plot_cumulative_distribution(AUM)
        sorted_indices = np.argsort(AUM)

        print(f'Removing {elbow_index} data samples that were identified to be misclassified.')
        removed_indices = sorted_indices[:elbow_index]
        retained_indices = np.setdiff1d(np.arange(len(self.dataset)), removed_indices)
        self.plot_removed_samples_distribution(removed_indices)
        self.visualize_lowest_aum_samples(removed_indices, AUM)

        if isinstance(self.dataset, AugmentedSubset):
            self.dataset.subset = torch.utils.data.Subset(self.dataset.subset, retained_indices)
        elif isinstance(self.dataset, IndexedDataset):
            self.dataset.dataset = torch.utils.data.Subset(self.dataset.dataset, retained_indices)
        else:
            raise ValueError("Dataset type not supported!")
