"""Core module for hardness-based resampling that returns the resampled dataset.

Important parts:
* Requires downloading the synthetic samples from https://github.com/wzekai99/DM-Improves-AT?tab=readme-ov-file. We use
the 1M version in our experiments, but the code should work for other versions as well (downloading will just take
longer). The downloaded data must be saved in the following format: GeneratedImages/{dataset_name}/{file_name}.npz
"""

from collections import defaultdict
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import TensorDataset, Subset
import torchvision

from src.config.config import ROOT
from src.data.datasets import AugmentedSubset, IndexedDataset
from src.hardness.estimators import compute_confidences
from src.models.loading import load_model_states


class DataResampling:
    """Class that contains all the methods required for hardness-based resampling"""
    def __init__(self, dataset: Union[AugmentedSubset, IndexedDataset], num_classes: int, oversampling_strategy: str,
                 undersampling_strategy: str, hardness_by_class: Dict[int, List[float]], high_is_hard: bool,
                 dataset_name: str, num_models_for_hardness: int, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float], num_epochs: int, holdout_set: Union[None, AugmentedSubset] = None):
        """Initialize the DataResampling class.

        :param dataset: The hardness-based resampling will be applied to this dataset
        :param num_classes: Number of classes in the dataset
        :param oversampling_strategy: Name of the oversampling strategy
        :param undersampling_strategy: Name of the undersampling strategy
        :param hardness_by_class: Instance-level hardness estimates divided by class
        :param high_is_hard: Set to true if high values of hardness estimate indicates hard samples (e.g., Loss)
        :param dataset_name: Name of the dataset
        :param num_models_for_hardness: Number of models from the trained ensemble that will be used to estimate
        hardness
        :param mean: The mean of the dataset used for normalization
        :param std: The std of the dataset used for normalization
        :param num_epochs: Amount of epochs used in training. This is used when performing EDM-based oversampling. For
        hEDM and aEDM we need to estimate hardness of synthetic data. We do it using Confidence of pretrained models.
        :param holdout_set: Contains the real data samples that were held out during pruning

        """
        self.dataset = dataset
        self.num_classes = num_classes
        self.oversampling_strategy = oversampling_strategy
        self.undersampling_strategy = undersampling_strategy
        self.hardness_by_class = hardness_by_class
        self.high_is_hard = high_is_hard
        self.dataset_name = dataset_name
        self.num_models_for_hardness = num_models_for_hardness
        self.mean = mean
        self.std = std
        self.num_epochs = num_epochs
        self.holdout_set = holdout_set

    def prune_easy(self, desired_count: int, hardness_scores: List[float]) -> List[int]:
        """Prune based on hardness focusing on the removal of easy samples.
        :returns: indices of the samples to keep after pruning"""
        sorted_indices = np.argsort(hardness_scores)
        if self.high_is_hard:
            return list(sorted_indices[-desired_count:])
        else:
            return list(sorted_indices[:desired_count])

    @staticmethod
    def random_oversample(desired_count: int, hardness_scores: List[float]) -> List[int]:
        """Perform random oversampling to match the desired count (we allow replacement)."""
        additional_indices = random.choices(range(len(hardness_scores)), k=desired_count - len(hardness_scores))
        return list(range(len(hardness_scores))) + additional_indices

    def SMOTE(self, oversample_target: int, current_indices: List[int], k: int = 5) -> torch.Tensor:
        """Perform oversampling using SMOTE to match the desired count."""
        current_n_samples, synthetic_samples = len(current_indices), []

        original_data_samples = torch.stack([self.dataset[idx][0] for idx in current_indices])
        original_data_samples_flattened = original_data_samples.view(current_n_samples, -1)

        neighbors = NearestNeighbors(n_neighbors=k + 1).fit(original_data_samples_flattened.numpy())
        _, neighbor_indices = neighbors.kneighbors(original_data_samples_flattened.numpy())

        for _ in range(oversample_target):
            idx = torch.randint(0, current_n_samples, (1,)).item()
            neighbor_idx = torch.randint(1, k + 1, (1,)).item()  # Skip the first neighbor (itself)

            sample = original_data_samples[idx]
            neighbor = original_data_samples[neighbor_indices[idx][neighbor_idx]]
            alpha = torch.rand(1).item()
            synthetic_sample = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic_sample)

        synthetic_samples = torch.stack(synthetic_samples)
        return synthetic_samples

    def EDM(self, oversample_targets: Dict[int, int], strategy: str) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa
        """Perform oversampling using synthetic samples generated via EDM (they need to be downloaded from
        https://github.com/wzekai99/DM-Improves-AT?tab=readme-ov-file and put in appropriate directory though)."""
        # Load the synthetic data generated by EDM
        synthetic_data = np.load(os.path.join(ROOT, 'GeneratedImages', f'{self.dataset_name}.npz'))
        synthetic_images = synthetic_data[synthetic_data.files[0]]
        to_tensor = torchvision.transforms.ToTensor()
        synthetic_images = torch.stack([to_tensor(img) for img in synthetic_images])
        synthetic_labels = synthetic_data[synthetic_data.files[1]]

        # Only continue with samples from the classes that we want to oversample
        synthetic_per_class, real_per_class = defaultdict(list), defaultdict(list)

        for i, label in enumerate(synthetic_labels):
            if label in oversample_targets.keys():
                synthetic_per_class[label].append(synthetic_images[i])
        for image, label, _ in self.dataset:
            if label.item() in oversample_targets:
                real_per_class[label].append(image)

        # Iterate through classes that we want to oversample
        all_synthetic_images, all_synthetic_labels = [], []
        for class_id in oversample_targets.keys():
            needed_count = oversample_targets[class_id]
            all_synthetic_labels.extend([class_id for _ in range(needed_count)])
            class_synthetic_images = synthetic_per_class[class_id]
            class_real_images = real_per_class[class_id]
            if strategy == 'random':
                selected_indices = random.sample(range(len(class_synthetic_images)), needed_count)
            else:
                model_states = load_model_states(self.dataset_name, self.num_models_for_hardness, self.num_epochs)
                average_synthetic_conf = compute_confidences(model_states, class_synthetic_images, class_id,
                                                             self.num_classes, self.mean, self.std)
                average_real_conf = np.mean(compute_confidences(model_states, class_real_images, class_id,
                                                                self.num_classes, self.mean, self.std))

                # We select the needed samples from a pool of samples to introduce randomness
                overshoot_count = min(2 * needed_count, len(class_synthetic_images))
                if strategy == 'hard':
                    sorted_indices = sorted(range(len(average_synthetic_conf)),
                                            key=lambda idx: average_synthetic_conf[idx])
                    indices_of_hardest_samples = sorted_indices[:overshoot_count]  # low confidence means hard samples
                    selected_indices = random.sample(indices_of_hardest_samples, needed_count)
                else:
                    relative_synthetic_confidences = abs(np.array(average_synthetic_conf) - average_real_conf)
                    sorted_indices = sorted(range(len(relative_synthetic_confidences)),
                                            key=lambda idx: relative_synthetic_confidences[idx])
                    indices_of_most_similar_samples = sorted_indices[:overshoot_count]
                    selected_indices = random.sample(indices_of_most_similar_samples, needed_count)
            selected_images = torch.stack([class_synthetic_images[i] for i in selected_indices])
            all_synthetic_images.append(selected_images)

        # Convert synthetic images to tensors
        all_synthetic_images = torch.cat(all_synthetic_images)
        all_synthetic_labels = torch.tensor(all_synthetic_labels)

        return all_synthetic_images, all_synthetic_labels

    def holdout_oversample(self, desired_count: int, class_id: int) -> torch.Tensor:
        """Perform random oversampling to match the desired count using the holdout set."""

        holdout_images = [self.holdout_set[idx][0] for idx in range(len(self.holdout_set))
                          if self.holdout_set[idx][1] == class_id]
        if len(holdout_images) < desired_count:
            raise ValueError(f"Holdout set only has {len(holdout_images)} samples for class {class_id}, need "
                             f"{desired_count}")

        resampled_holdout_images, resampled_holdout_labels = [], []
        selected_indices = random.sample(range(len(holdout_images)), desired_count)
        selected_images = torch.stack([holdout_images[i] for i in selected_indices])
        resampled_holdout_images.append(selected_images)
        resampled_holdout_labels.extend([class_id for _ in range(desired_count)])

        synthetic_images = torch.cat(resampled_holdout_images)

        return synthetic_images

    def select_oversampling_method(self) -> Optional[Callable[..., Any]]:
        """
        Select the appropriate oversampling method based on the strategy.

        Returns:
            Callable[..., Any]: A function implementing the oversampling strategy.
            None: If strategy == "none".
        Raises:
            ValueError: If the strategy is unsupported. Shouldn't happen due to earlier precautions (sanity check).
        """
        if self.oversampling_strategy == "random":
            return lambda desired_count, hardness: self.random_oversample(desired_count, hardness)
        elif self.oversampling_strategy == 'SMOTE':
            return lambda oversample_target, current_indices: self.SMOTE(oversample_target, current_indices)
        elif self.oversampling_strategy == 'rEDM':
            return lambda oversample_targets: self.EDM(oversample_targets, 'random')
        elif self.oversampling_strategy == 'hEDM':
            return lambda oversample_targets: self.EDM(oversample_targets, 'hard')
        elif self.oversampling_strategy == 'aEDM':
            return lambda oversample_targets: self.EDM(oversample_targets, 'average')
        elif self.oversampling_strategy == 'holdout':
            return lambda oversample_target, class_id: self.holdout_oversample(oversample_target, class_id)
        elif self.oversampling_strategy == 'none':
            return None
        else:
            raise ValueError(f"Oversampling strategy {self.oversampling_strategy} is not supported.")

    def resample_data(self, samples_per_class: List[int]):
        """
        Perform resampling to match the desired samples_per_class. Uses the selected undersampling and oversampling
        methods.
        ------------------------------------------------------------------------------------------------
        There are two types of oversampling strategies - ones that generate new data and one that reuse existing data.
        """
        # Select oversampling method
        oversample = self.select_oversampling_method()

        # Organize dataset by classes
        class_indices = {i: [] for i in range(self.num_classes)}
        for _, label, idx in self.dataset:
            class_indices[label.item()].append(idx)

        resampled_indices, synthetic_data, synthetic_labels, desired_counts, current_counts = [], [], [], [], []

        # Perform resampling for each class
        for class_id, hardnesses_within_class in self.hardness_by_class.items():
            desired_count = samples_per_class[class_id]
            desired_counts.append(desired_count)
            current_indices = class_indices[class_id]
            current_counts.append(len(current_indices))

            if len(current_indices) > desired_count and self.undersampling_strategy != 'none':
                class_retain_indices = self.prune_easy(desired_count, hardnesses_within_class)
                resampled_indices.extend(np.array(current_indices)[class_retain_indices])
            elif len(current_indices) < desired_count and self.oversampling_strategy != 'none':
                n_samples_to_produce = desired_count - len(current_indices)
                if self.oversampling_strategy == 'SMOTE':
                    synthetic_data.append(oversample(n_samples_to_produce, current_indices))
                    synthetic_labels.append(torch.full((n_samples_to_produce,), class_id))

                    resampled_indices.extend(current_indices)
                elif self.oversampling_strategy == 'random':
                    class_add_indices = oversample(desired_count, hardnesses_within_class)
                    resampled_indices.extend(np.array(current_indices)[class_add_indices])
                elif self.oversampling_strategy == 'holdout':
                    synthetic_data.append(oversample(n_samples_to_produce, class_id))
                    synthetic_labels.append(torch.full((n_samples_to_produce,), class_id))

                    resampled_indices.extend(current_indices)
                else:
                    resampled_indices.extend(current_indices)
            elif len(current_indices) == desired_count:
                resampled_indices.extend(current_indices)

        # We do this outside loop as loading the EDM samples is time-consuming.
        if self.oversampling_strategy in ['rEDM', 'hEDM', 'aEDM']:
            oversample_targets = {
                class_id: desired_counts[class_id] - current_counts[class_id]
                for class_id in range(self.num_classes)
                if desired_counts[class_id] > current_counts[class_id]
            }
            synthetic_data, synthetic_labels = oversample(oversample_targets)

        elif self.oversampling_strategy in ['SMOTE', 'holdout']:
            synthetic_data = torch.cat(synthetic_data)
            synthetic_labels = torch.cat(synthetic_labels)

        if self.oversampling_strategy in ['SMOTE', 'rEDM', 'hEDM', 'aEDM', 'holdout']:
            original_data_samples = torch.stack([self.dataset[idx][0] for idx in range(len(self.dataset))])
            original_labels = torch.stack([self.dataset[idx][1] for idx in range(len(self.dataset))])

            new_data = torch.cat([original_data_samples, synthetic_data])
            new_labels = torch.cat([original_labels, synthetic_labels])
            self.dataset = IndexedDataset(TensorDataset(new_data, new_labels))

            synthetic_data_start_idx = original_data_samples.size(0)
            synthetic_indices = list(range(synthetic_data_start_idx, new_data.size(0)))
            resampled_indices.extend(synthetic_indices)

        return AugmentedSubset(Subset(self.dataset, resampled_indices))
