import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from utils import get_config


class DataPruning:
    def __init__(self, instance_hardness: List[List[float]], class_hardness: Dict[int, List[List[float]]],
                 labels: List[int], prune_percentage: int, dataset_name: str, protect_prototypes: bool):
        """
        Initialize the DataPruning class.

        :param instance_hardness: List of lists of instance-level hardness scores.
        :param class_hardness: Dictionary where each key is a class and the value is a list of lists of class-level
        hardness scores.
        :param labels: List of class labels corresponding to each instance.
        :param prune_percentage: Percentage of the data to prune (default is 50%).
        :param dataset_name: Name of the dataset we are working on (used for saving),
        :param protect_prototypes: If set to true, the pruning will omit the 1% of the easiest data samples
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
        self.protect_prototypes = protect_prototypes
        self.fig_save_dir = 'Figures/'
        self.res_save_dir = 'Results/'
        self.num_classes = get_config(dataset_name)['num_classes']

        # Load or initialize class_level_sample_counts
        try:
            with open("class_level_sample_counts.pkl", "rb") as file:
                self.class_level_sample_counts = pickle.load(file)
        except (FileNotFoundError, EOFError):
            self.class_level_sample_counts = {}

    def get_prune_indices(self, hardness_scores, retain_count):
        sorted_indices = np.argsort(hardness_scores)

        if self.protect_prototypes:
            # Protect the easiest 1% samples
            one_percent = int(0.01 * len(hardness_scores))
            return sorted_indices[one_percent:one_percent + retain_count]
        else:
            return sorted_indices[-retain_count:]

    def plot_class_level_sample_distribution(self, remaining_indices: List[int], pruning_key: str):
        os.makedirs(self.fig_save_dir, exist_ok=True)
        os.makedirs(self.res_save_dir, exist_ok=True)

        # Count the number of remaining samples for each class
        remaining_labels = self.labels[remaining_indices]
        unique_classes, class_counts = np.unique(remaining_labels, return_counts=True)

        # Create dictionary for current pruning type if it doesn't exist
        if pruning_key not in self.class_level_sample_counts:
            self.class_level_sample_counts[pruning_key] = {}

        # Store the distribution of samples after pruning
        self.class_level_sample_counts[pruning_key][int(self.prune_percentage * 100)] = [
            class_counts[unique_classes.tolist().index(cls)] if cls in unique_classes else 0
            for cls in range(self.num_classes)
        ]

        # Save the updated class_level_sample_counts to a pickle file
        with open(os.path.join(self.res_save_dir, "class_level_sample_counts.pkl"), "wb") as file:
            pickle.dump(self.class_level_sample_counts, file)

        # Plot the original class distribution
        plt.figure()
        plt.bar(unique_classes, class_counts)
        plt.ylabel('Number of Remaining Samples')
        plt.title(f'Class-level Distribution of Remaining Samples After {pruning_key.upper()} Pruning')
        plt.xticks([])
        plt.savefig(os.path.join(self.fig_save_dir, 'class_level_sample_distribution.pdf'))
        plt.close()

        # Sort classes by class_counts for imbalance visualization
        sorted_indices = np.argsort(class_counts)
        sorted_classes = unique_classes[sorted_indices]
        sorted_counts = class_counts[sorted_indices]

        # Plot sorted class distribution to highlight data imbalance
        plt.figure()
        plt.bar(range(len(sorted_classes)), sorted_counts)
        plt.ylabel('Number of Remaining Samples')
        plt.title(f'Sorted Class-level Distribution of Remaining Samples After {pruning_key.upper()} Pruning')
        plt.xticks([])
        plt.savefig(os.path.join(self.fig_save_dir, 'sorted_class_level_sample_distribution.pdf'))
        plt.close()

    def dataset_level_pruning(self):
        """
        Remove the specified percentage of samples with the lowest instance-level hardness.
        This method returns the indices of the remaining data after pruning.

        :return: List of indices of the remaining data samples after pruning.
        """
        retain_count = int((1 - self.prune_percentage) * len(self.instance_hardness))
        remaining_indices = self.get_prune_indices(self.instance_hardness, retain_count)

        self.fig_save_dir = os.path.join(self.fig_save_dir, 'dlp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.res_save_dir = os.path.join(self.res_save_dir, 'dlp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.plot_class_level_sample_distribution(remaining_indices.tolist(), pruning_key='dlp')

        return remaining_indices.tolist()


    def fixed_class_level_pruning(self):
        """
        Remove the specified percentage of samples from each class.
        Ensures that the class distribution remains balanced.

        :return: List of indices of the remaining data samples after class-level pruning.
        """
        remaining_indices = []

        for class_id, class_scores in self.class_hardness.items():
            retain_count = int((1 - self.prune_percentage) * len(class_scores))
            class_remaining_indices = self.get_prune_indices(class_scores, retain_count)
            global_indices = np.where(self.labels == class_id)[0]
            remaining_indices.extend(global_indices[class_remaining_indices])

        self.fig_save_dir = os.path.join(self.fig_save_dir, 'fclp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.res_save_dir = os.path.join(self.res_save_dir, 'fclp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.plot_class_level_sample_distribution(remaining_indices, pruning_key='fclp')

        return remaining_indices

    def adaptive_class_level_pruning(self, scaling_type="linear", alpha=3):
        """
        Perform adaptive class-level pruning based on the average hardness of each class.
        Harder classes will lose fewer samples, while easier classes will lose more.
        Scaling type can be 'linear', 'exponential', or 'inverted_exponential'.

        :param scaling_type: Type of scaling ('linear', 'exponential', or 'inverted_exponential').
        :param alpha: Parameter for exponential scaling to control fairness.
        :return: List of indices of the remaining data samples after adaptive class-level pruning.
        """
        remaining_indices, epsilon = [], 1e-6

        # Calculate the mean hardness of each class
        class_mean_hardness = {class_id: np.mean(class_scores) for class_id, class_scores in
                               self.class_hardness.items()}

        # Get the min and max class hardness values
        max_hardness, min_hardness = max(class_mean_hardness.values()), min(class_mean_hardness.values())

        # Normalize hardness values to range from 0 to 1
        hardness_range = max_hardness - min_hardness
        if hardness_range == 0:
            hardness_range = epsilon  # Avoid division by zero
        normalized_class_mean_hardness = {
            class_id: (mean_hardness - min_hardness) / hardness_range
            for class_id, mean_hardness in class_mean_hardness.items()
        }

        # Iterate over each class in class_hardness
        for class_id, class_scores in self.class_hardness.items():
            class_sample_count = len(class_scores)
            normalized_mean_hardness = normalized_class_mean_hardness[class_id]

            # Calculate scaling factor based on chosen scaling type
            if scaling_type == "linear":
                scaling_factor = normalized_mean_hardness
            elif scaling_type == "exponential":
                scaling_factor = (np.exp(alpha * normalized_mean_hardness) - 1) / (np.exp(alpha) - 1)
            elif scaling_type == "inverted_exponential":
                scaling_factor = (1 - np.exp(-alpha * normalized_mean_hardness)) / (1 - np.exp(-alpha))
            else:
                raise ValueError("Unsupported scaling type. Choose 'linear', 'exponential', or 'inverted_exponential'.")

            # Calculate class pruning percentage
            class_prune_percentage = self.prune_percentage * (1 - scaling_factor)
            retain_count = int((1 - class_prune_percentage) * class_sample_count)

            # Get adjusted prune indices considering protect_prototypes
            class_remaining_indices = self.get_prune_indices(class_scores, retain_count)

            # Find global indices that belong to this class using the labels
            global_indices = np.where(self.labels == class_id)[0]
            remaining_indices.extend(global_indices[class_remaining_indices])

        pruning_key = f"{scaling_type}_aclp"
        self.fig_save_dir = os.path.join(self.fig_save_dir, f'{pruning_key}{int(self.prune_percentage * 100)}',
                                         self.dataset_name)
        self.res_save_dir = os.path.join(self.res_save_dir, f'{pruning_key}{int(self.prune_percentage * 100)}',
                                         self.dataset_name)
        self.plot_class_level_sample_distribution(remaining_indices, pruning_key=pruning_key)

        return remaining_indices

    def leave_one_out_pruning(self):
        """
        Perform leave-one-out pruning, where all classes are pruned equally based on the prune_percentage,
        except for the hardest class, which is not pruned.

        :return: List of indices of the remaining data samples after leave-one-out pruning.
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
            retain_count = class_sample_count if class_id == hardest_class_id else int(
                (1 - self.prune_percentage) * class_sample_count)

            # Get adjusted prune indices considering protect_prototypes
            class_remaining_indices = self.get_prune_indices(class_scores, retain_count)

            # Find global indices that belong to this class using the labels
            global_indices = np.where(self.labels == class_id)[0]
            remaining_indices.extend(global_indices[class_remaining_indices])

        self.fig_save_dir = os.path.join(self.fig_save_dir, 'loop' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.res_save_dir = os.path.join(self.res_save_dir, 'loop' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.plot_class_level_sample_distribution(remaining_indices, pruning_key='loop')

        return remaining_indices
