import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


class DataPruning:
    def __init__(self, instance_hardness: List[List[float]], class_hardness: Dict[int, List[List[float]]],
                 labels: List[int], prune_percentage: float = 0.5, dataset_name: str = 'MNIST'):
        """
        Initialize the DataPruning class.

        :param instance_hardness: List of lists of instance-level hardness scores.
        :param class_hardness: Dictionary where each key is a class and the value is a list of lists of class-level
        hardness scores.
        :param labels: List of class labels corresponding to each instance.
        :param prune_percentage: Fraction of the data to prune (default is 0.5, i.e., 50%).
        :param dataset_name: Name of the dataset we are working on (used for saving)
        """
        # Compute the average instance-level hardness for each sample across all models
        self.instance_hardness = np.mean(np.array(instance_hardness), axis=1)
        # Compute the average class-level hardness for each class across all models
        self.class_hardness = {
            class_id: np.mean(np.array(class_scores), axis=1)
            for class_id, class_scores in class_hardness.items()
        }
        self.labels = np.array(labels)
        self.prune_percentage = prune_percentage
        self.dataset_name = dataset_name

    def plot_class_level_sample_distribution(self, remaining_indices: List[int], file_name: str):
        """
        Plot the distribution of remaining samples across different classes after pruning.

        :param remaining_indices: List of indices of the remaining data samples after pruning.
        :param file_name: Name of the file used for saving the Figure.
        """
        # Get the remaining labels using the indices
        remaining_labels = self.labels[remaining_indices]

        # Count the number of remaining samples for each class
        unique_classes, class_counts = np.unique(remaining_labels, return_counts=True)

        # Plot the class distribution
        plt.bar(unique_classes, class_counts)
        plt.xlabel('Class ID')
        plt.ylabel('Number of Remaining Samples')
        plt.title('Class-level Distribution of Remaining Samples After Pruning')

        os.makedirs('Figures/', exist_ok=True)
        plt.savefig(file_name)
        plt.show()

    def dataset_level_pruning(self):
        """
        Remove the specified percentage of samples with the lowest instance-level hardness.
        This method returns the indices of the remaining data after pruning.

        :return: List of indices of the remaining data samples after pruning.
        """
        # Number of samples to retain after pruning
        retain_count = int((1 - self.prune_percentage) * len(self.instance_hardness))

        # Get the indices that would sort the array in ascending order of hardness
        sorted_indices = np.argsort(self.instance_hardness)

        # Retain the top 'retain_count' number of the hardest samples
        remaining_indices = sorted_indices[-retain_count:]

        # Plot the class distribution after pruning
        self.plot_class_level_sample_distribution(remaining_indices.tolist(), f'Figures/dd_dlp{self.dataset_name}.pdf')

        return remaining_indices.tolist()

    def fixed_class_level_pruning(self):
        """
        Remove the specified percentage (50% by default) of samples from each class.
        This ensures that the class distribution remains balanced.

        :return: List of indices of the remaining data samples after class-level pruning.
        """
        remaining_indices = []

        # Iterate over each class in class_hardness
        for class_id, class_scores in self.class_hardness.items():
            class_sample_count = len(class_scores)  # Number of samples in this class
            retain_count = int((1 - self.prune_percentage) * class_sample_count)

            # Get the indices that would sort the class-level hardness for this class
            sorted_class_indices = np.argsort(class_scores)

            # Retain the top 'retain_count' number of the hardest samples from this class
            class_remaining_indices = sorted_class_indices[-retain_count:]

            # Find global indices that belong to this class using the labels
            global_indices = np.where(self.labels == class_id)[0]
            remaining_indices.extend(global_indices[class_remaining_indices])

        # Plot the class distribution after pruning
        self.plot_class_level_sample_distribution(remaining_indices, f'Figures/dd_fclp{self.dataset_name}.pdf')

        return remaining_indices
