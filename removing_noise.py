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

    @staticmethod
    def load_pickle(file_path):
        """Load data from a pickle file."""
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def plot_cumulative_distribution(data, title="Cumulative Distribution with Elbow"):
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

        # Compute the value at x = 2.5
        x_value = 2.5
        y_value = np.interp(x_value, sorted_data, cumulative_percentage)
        elbow_index = np.searchsorted(sorted_data, x_value, side='left')
        plt.axhline(y=y_value, color='red', linestyle='--', label=f'Value at x = {x_value}')
        plt.scatter([x_value], [y_value], color='red', zorder=5)  # Label the point for clarity
        plt.legend()
        plt.savefig('AUM_distribution.pdf')

        return elbow_index

    @staticmethod
    def visualize_lowest_aum_samples(data, labels, indices, title="Lowest AUM Samples"):
        """
        Visualizes the 10 samples with the lowest AUM values and their respective classes.

        Args:
            data (np.ndarray): The data samples.
            labels (np.ndarray): Corresponding labels.
            indices (np.ndarray): Indices of the lowest AUM samples.
            title (str): Title of the plot.
        """
        lowest_indices = indices[:10]
        lowest_samples = data[lowest_indices]
        lowest_labels = labels[lowest_indices]

        plt.figure(figsize=(15, 6))
        for i, (sample, label) in enumerate(zip(lowest_samples, lowest_labels)):
            plt.subplot(2, 5, i + 1)
            plt.imshow(sample, cmap='gray')  # Adjust cmap based on the dataset type (grayscale or RGB)
            plt.title(f"Class: {label}")
            plt.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig('label_noise.pdf')

    def clean(self):
        file = f"Results/{self.dataset_name}/AUM.pkl"
        AUM_over_epochs = np.array(self.load_pickle(file))
        AUM = np.mean(AUM_over_epochs, axis=1)

        elbow_index = self.plot_cumulative_distribution(AUM)
        sorted_indices = np.argsort(AUM)

        print(f'Removing {elbow_index} data samples that were identified to be misclassified.')
        remove_indices = sorted_indices[:elbow_index]
        retained_indices = np.setdiff1d(np.arange(len(self.dataset)), remove_indices)
        if isinstance(self.dataset, AugmentedSubset):
            self.dataset.subset = torch.utils.data.Subset(self.dataset.subset, retained_indices)
        elif isinstance(self.dataset, IndexedDataset):
            self.dataset.dataset = torch.utils.data.Subset(self.dataset.dataset, retained_indices)
        else:
            raise ValueError("Dataset type not supported!")

        all_data, all_labels, _ = zip(*[self.dataset[i] for i in range(len(self.dataset))])
        self.visualize_lowest_aum_samples(
            data=np.array(all_data),
            labels=np.array(all_labels),
            indices=sorted_indices
        )
