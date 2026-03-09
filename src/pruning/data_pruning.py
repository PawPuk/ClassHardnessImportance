"""Core module for pruning the dataset that returns the pruned subdataset."""

from typing import Dict, List, Tuple, Union

import numpy as np
from torch.utils.data import Subset

from src.config.config import get_config
from src.data.datasets import AugmentedSubset, IndexedDataset
from src.resampling.data_resampling import DataResampling


class DataPruning:
    """Class that contains all the methods required for producing the pruned subdatasets. It outputs the pruned
    subdataset."""
    def __init__(
            self,
            prune_percentage: int,
            dataset_name: str,
            samples_per_class: List[int],
            hardness_estimator: str
    ):
        """
        Initialize the DataPruning class.

        :param prune_percentage: Percentage of the data to prune.
        :param dataset_name: Name of the dataset we are working on (used for saving).
        :param samples_per_class: List of integers specifying the number of samples to be left in each class after
        hardness-based resampling - for resampling_pruned_subdataset().
        :param hardness_estimator: Name of the used hardness estimator
        """
        self.prune_percentage = prune_percentage / 100
        self.dataset_name = dataset_name
        self.imbalance_ratio = samples_per_class
        self.hardness_estimator = hardness_estimator

        config = get_config(dataset_name)
        self.num_classes = config['num_classes']
        self.num_samples_per_class = config['num_training_samples']
        self.num_models_for_hardness = config['num_models_for_hardness']
        self.num_models_per_dataset = config['num_models_per_dataset']
        self.mean = config['mean']
        self.std = config['std']
        self.num_epochs = config['num_epochs']

    def class_level_pruning(
            self,
            labels: List[int],
            training_dataset: Union[AugmentedSubset, IndexedDataset],
            hardness_by_class: Dict[int, List[Tuple[int, float]]]
    ) -> Tuple[AugmentedSubset, AugmentedSubset, Dict[int, List[float]]]:
        """
        Remove the specified percentage of samples from each class, ensuring that the class distribution remains
        balanced.

        :param labels: List of labels for each data sample in the original dataset.
        :param training_dataset: Original training dataset
        :param hardness_by_class: Instance-level hardness estimates (and the corresponding sample indices) sorted
        by class. We will have to remove the estimates of samples that will get pruned.

        :return: Balanced subdataset, containing the remaining samples, the pruned_dataset containing the pruned
        samples, and the updated hardness estimates.
        """
        subdataset_indices = []

        for class_id in range(self.num_classes):
            retain_count = int((1 - self.prune_percentage) * self.num_samples_per_class[class_id])
            class_remaining_indices = np.random.choice(self.num_samples_per_class[class_id], retain_count,
                                                       replace=False)
            global_indices = np.where(np.array(labels) == class_id)[0]
            # We sort the indices because the hardness scores are sorted by indices, and we need those two to match.
            subdataset_indices.extend(np.sort(global_indices[class_remaining_indices]))
        subdataset = AugmentedSubset(Subset(training_dataset, subdataset_indices))
        # Update the hardness estimates of subdataset by removing the estimates of pruned samples
        hardness_by_class = {c: [hardness_by_class[c][i][1] for i in range(len(hardness_by_class[c]))
                                 if hardness_by_class[c][i][0] in subdataset_indices]
                             for c in hardness_by_class.keys()}

        # Create the pruned dataset for use in resampling as holdout set
        pruned_indices = list(set(range(len(training_dataset))) - set(subdataset_indices))
        pruned_dataset = AugmentedSubset(Subset(training_dataset, pruned_indices))

        return subdataset, pruned_dataset, hardness_by_class

    def prune_and_resample(
            self,
            oversampling_strategy: str,
            labels: List[int],
            training_dataset: Union[AugmentedSubset, IndexedDataset],
            hardness_by_class: Dict[int, List[Tuple[int, float]]]
    ) -> AugmentedSubset:
        """
        Removes the specified percentage of samples from the dataset. Produces imbalanced subdatasets. Works by firstly
        performing class_Level_pruning, and later performing hardness_based resampling on the pruned subdataset to
        introduce the data imbalance. Uses easy pruning as a default undersampling strategy, unless
        oversampling_strategy is "none" in which case no resampling is applied.

        :param oversampling_strategy: Name of the oversampling strategy for hardness-based resampling. If set to "none"
        then no resampling is applied.
        :param labels: List of labels for each data sample in the original dataset.
        :param training_dataset: Original training dataset
        :param hardness_by_class: Instance-level hardness estimates together with corresponding sample indices
        used to guide pruning. If these are set to None then EL2N scores are computed from the pruned subdatasets.

        :return: Imbalanced pruned subdataset
        """
        subdataset, pruned_dataset, hardness_by_class = self.class_level_pruning(labels, training_dataset,
                                                                                 hardness_by_class)

        if oversampling_strategy != "none":
            high_is_hard = self.hardness_estimator not in ['Confidence', 'AUM']
            resampler = DataResampling(subdataset, self.num_classes, oversampling_strategy, 'easy',
                                       hardness_by_class, high_is_hard, self.dataset_name, self.num_models_for_hardness,
                                       self.mean, self.std, self.num_epochs, pruned_dataset)
            subdataset = resampler.resample_data(self.imbalance_ratio)

        return subdataset
