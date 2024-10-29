import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


class DataPruning:
    def __init__(self, instance_hardness: List[List[float]], class_hardness: Dict[int, List[List[float]]],
                 labels: List[int], prune_percentage: int, dataset_name: str):
        """
        Initialize the DataPruning class.

        :param instance_hardness: List of lists of instance-level hardness scores.
        :param class_hardness: Dictionary where each key is a class and the value is a list of lists of class-level
        hardness scores.
        :param labels: List of class labels corresponding to each instance.
        :param prune_percentage: Percentage of the data to prune (default is 50%).
        :param dataset_name: Name of the dataset we are working on (used for saving)
        """
        # Compute the average instance-level hardness for each sample across all models
        self.instance_hardness = np.mean(np.array(instance_hardness), axis=1)
        # Compute the average class-level hardness for each sample in specific class across all models
        self.class_hardness = {
            class_id: np.mean(np.array(class_scores), axis=1)
            for class_id, class_scores in class_hardness.items()
        }
        self.labels = np.array(labels)
        self.prune_percentage = prune_percentage / 100
        self.dataset_name = dataset_name
        self.save_dir = 'Figures/'

    def plot_class_level_sample_distribution(self, remaining_indices: List[int]):
        """
        Plot the distribution of remaining samples across different classes after pruning.

        :param remaining_indices: List of indices of the remaining data samples after pruning.
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

        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(os.path.join(self.save_dir, 'class_level_sample_distribution.pdf'))

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
        self.save_dir = os.path.join(self.save_dir, 'dlp' + str(self.prune_percentage), self.dataset_name)
        self.plot_class_level_sample_distribution(remaining_indices.tolist())

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
        self.save_dir = os.path.join(self.save_dir, 'fclp' + str(self.prune_percentage), self.dataset_name)
        self.plot_class_level_sample_distribution(remaining_indices)

        return remaining_indices

    def random_pruning(self):
        """
        Perform random pruning by removing the specified percentage of samples randomly.

        :return: List of indices of the remaining data samples after random pruning.
        """
        # Number of samples to retain after pruning
        total_samples = len(self.labels)
        retain_count = int((1 - self.prune_percentage) * total_samples)

        # Randomly select indices to retain
        all_indices = np.arange(total_samples)
        remaining_indices = np.random.choice(all_indices, size=retain_count, replace=False)

        # Plot the class distribution after random pruning
        self.save_dir = os.path.join(self.save_dir, 'rp' + str(self.prune_percentage), self.dataset_name)
        self.plot_class_level_sample_distribution(remaining_indices.tolist())

        return remaining_indices.tolist()

    def adaptive_class_level_pruning(self, scaling_type="linear"):
        """
        Perform adaptive class-level pruning based on the average hardness of each class.
        Harder classes will lose fewer samples, while easier classes will lose more.
        Scaling type can be 'linear', 'exponential', or 'logarithmic'.

        :param scaling_type: Type of scaling ('linear', 'exponential', 'logarithmic'). Default is 'linear'.
        :return: List of indices of the remaining data samples after adaptive class-level pruning.
        """
        remaining_indices, epsilon = [], 1e-6

        # Calculate the mean hardness of each class
        class_mean_hardness = {class_id: np.mean(class_scores) for class_id, class_scores in
                               self.class_hardness.items()}

        # Get the min and max class hardness values
        max_hardness, min_hardness = max(class_mean_hardness.values()), min(class_mean_hardness.values())

        # Iterate over each class in class_hardness
        for class_id, class_scores in self.class_hardness.items():
            class_sample_count = len(class_scores)
            mean_hardness = class_mean_hardness[class_id]

            # Scale pruning rate based on class hardness (hardest class keeps all, easiest prunes at prune_percentage)
            if max_hardness != min_hardness:
                # Calculate scaling factor based on chosen scaling type
                if scaling_type == "linear":
                    scaling_factor = (mean_hardness - min_hardness) / (max_hardness - min_hardness)
                elif scaling_type == "exponential":
                    scaling_factor = np.exp(mean_hardness - min_hardness) / np.exp(max_hardness - min_hardness)
                elif scaling_type == "logarithmic":
                    scaling_factor = np.log(mean_hardness - min_hardness + epsilon) / np.log(
                        max_hardness - min_hardness + epsilon)
                else:
                    raise ValueError("Unsupported scaling type. Choose 'linear', 'exponential', or 'logarithmic'.")
            else:
                scaling_factor = 1  # If all classes have the same hardness, treat all equally (no scaling)

            class_prune_percentage = self.prune_percentage * (1 - scaling_factor)
            retain_count = int((1 - class_prune_percentage) * class_sample_count)

            # Get the indices that would sort the class-level hardness for this class
            sorted_class_indices = np.argsort(class_scores)

            # Retain the top 'retain_count' number of the hardest samples from this class
            class_remaining_indices = sorted_class_indices[-retain_count:]

            # Find global indices that belong to this class using the labels
            global_indices = np.where(self.labels == class_id)[0]
            remaining_indices.extend(global_indices[class_remaining_indices])

        # Plot the class distribution after pruning
        self.save_dir = os.path.join(self.save_dir, f'{scaling_type}_aclp' + str(self.prune_percentage), self.dataset_name)
        self.plot_class_level_sample_distribution(remaining_indices)

        return remaining_indices

    def live_one_out_pruning(self):
        """
        Perform live-out-out pruning, where all classes are pruned equally based on the prune_percentage,
        except for the hardest class, which is not pruned.

        :return: List of indices of the remaining data samples after live-out-out pruning.
        """
        remaining_indices = []

        # Calculate the mean hardness of each class
        class_mean_hardness = {class_id: np.mean(class_scores) for class_id, class_scores in 
                               self.class_hardness.items()}

        # Find the hardest class (with the maximum average hardness)
        hardest_class_id = max(class_mean_hardness, key=class_mean_hardness.get)

        # Iterate over each class in class_hardness
        for class_id, class_scores in self.class_hardness.items():
            class_sample_count = len(class_scores)

            if class_id == hardest_class_id:
                # Don't prune the hardest class, retain all samples
                retain_count = class_sample_count
            else:
                # Prune according to the prune_percentage for other classes
                retain_count = int((1 - self.prune_percentage) * class_sample_count)

            # Get the indices that would sort the class-level hardness for this class
            sorted_class_indices = np.argsort(class_scores)

            # Retain the top 'retain_count' number of the hardest samples from this class
            class_remaining_indices = sorted_class_indices[-retain_count:]

            # Find global indices that belong to this class using the labels
            global_indices = np.where(self.labels == class_id)[0]
            remaining_indices.extend(global_indices[class_remaining_indices])

        # Plot the class distribution after pruning
        self.save_dir = os.path.join(self.save_dir, 'loolp' + str(self.prune_percentage), self.dataset_name)
        self.plot_class_level_sample_distribution(remaining_indices)

        return remaining_indices


