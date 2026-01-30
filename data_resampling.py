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

from config import DEVICE, ROOT
from data import AugmentedSubset, IndexedDataset
from neural_networks import ResNet18LowRes


class DataResampling:
    """Class that contains all the methods required for hardness-based resampling"""
    def __init__(self, dataset: Union[AugmentedSubset, IndexedDataset], num_classes: int, oversampling_strategy: str,
                 undersampling_strategy: str, hardness_by_class: Dict[int, List[float]], high_is_hard: bool,
                 dataset_name: str, num_models_for_hardness: int, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float], holdout_set: Union[None, AugmentedSubset] = None):
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
        self.holdout_set = holdout_set

    def prune_easy(self, desired_count: int, hardness_scores: List[float]) -> List[int]:
        """Prune based on hardness focusing on the removal of easy samples."""
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

    def SMOTE(self, desired_count: int, current_indices: List[int], k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa
        """Perform oversampling using SMOTE to match the desired count."""
        current_n_samples, synthetic_samples = len(current_indices), []

        original_data_samples = torch.stack([self.dataset[idx][0] for idx in current_indices])
        original_data_samples_flattened = original_data_samples.view(current_n_samples, -1)

        neighbors = NearestNeighbors(n_neighbors=k + 1).fit(original_data_samples_flattened.numpy())
        _, neighbor_indices = neighbors.kneighbors(original_data_samples_flattened.numpy())

        for _ in range(desired_count - current_n_samples):
            idx = torch.randint(0, current_n_samples, (1,)).item()
            neighbor_idx = torch.randint(1, k + 1, (1,)).item()  # Skip the first neighbor (itself)

            sample = original_data_samples[idx]
            neighbor = original_data_samples[neighbor_indices[idx][neighbor_idx]]
            alpha = torch.rand(1).item()
            synthetic_sample = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic_sample)

        synthetic_samples = torch.stack(synthetic_samples)
        return original_data_samples, synthetic_samples

    def load_model_states(self) -> List[Any]:
        """Load the pre-trained models."""
        models_dir = os.path.join(ROOT, "Models")
        model_states = []
        full_dataset_dir = os.path.join(models_dir, "none", f"unclean{self.dataset_name}")

        for file in os.listdir(full_dataset_dir):
            if len(model_states) < self.num_models_for_hardness and file.endswith(".pth") and "_epoch_200" in file:
                model_path = os.path.join(full_dataset_dir, file)
                model_state = torch.load(model_path)
                model_states.append(model_state)

        print(f"Loaded {len(model_states)} models for estimating confidence.")
        return model_states

    def compute_confidences(self, model_states: List[Any], images: List[torch.Tensor], class_id: int,
                            batch_size: int = 1024) -> List[float]:
        """For a given class_id, compute average confidence across models for each image."""
        num_samples, avg_confidences = len(images), []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_images = images[batch_start:batch_end]

            normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
            normalized_images = [normalize(img) for img in batch_images]
            batch_normalized_images = torch.stack(normalized_images).to(DEVICE)  # Shape: [B, 3, 32, 32]

            # For each model, compute confidence
            batch_confidences = torch.zeros(batch_normalized_images.size(0), device=DEVICE)
            for model_state in model_states:
                model = ResNet18LowRes(self.num_classes)
                model.load_state_dict(model_state)
                model = model.to(DEVICE)
                model.eval()
                with torch.no_grad():
                    logits = model(batch_normalized_images)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    conf = probs[:, class_id]  # confidence for true class
                    batch_confidences += conf  # accumulate per model

            batch_confidences /= len(model_states)  # average confidence across models
            avg_confidences.extend(batch_confidences.cpu().tolist())

        return avg_confidences

    def extract_original_data_and_labels(self, oversample_targets: Dict[int, int]
                                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract the original data samples and labels from classes that we want to perform oversampling on. We will
        later add the oversampled samples onto these samples to create the final resampled dataset."""
        original_data, original_labels = [], []
        for idx in range(len(self.dataset)):
            image, label, _ = self.dataset[idx]
            if label.item() in oversample_targets:
                original_data.append(image)
                original_labels.append(label)
        original_data = torch.stack(original_data)
        original_labels = torch.stack(original_labels)
        return original_data, original_labels

    def EDM(self, oversample_targets: Dict[int, int], strategy: str) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa
        """Perform oversampling using synthetic samples generated via EDM (they need to be downloaded from
        https://github.com/wzekai99/DM-Improves-AT?tab=readme-ov-file and put in appropriate directory though)."""
        # Load the synthetic data generated by EDM
        synthetic_data = np.load(os.path.join(ROOT, 'GeneratedImages', f'{self.dataset_name}.npz'))
        synthetic_images = synthetic_data[synthetic_data.files[0]]  # numpy array of arrays of ...
        to_tensor = torchvision.transforms.ToTensor()
        synthetic_images = torch.stack([to_tensor(img) for img in synthetic_images])
        synthetic_labels = synthetic_data[synthetic_data.files[1]]  # numpy array of integers

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
                model_states = self.load_model_states()
                average_synthetic_confidences = self.compute_confidences(model_states, class_synthetic_images, class_id)
                average_real_confidence = np.mean(self.compute_confidences(model_states, class_real_images, class_id))
                if strategy == 'hard':
                    sorted_indices = sorted(range(len(average_synthetic_confidences)),
                                            key=lambda idx: average_synthetic_confidences[idx])
                    indices_of_hardest_samples = sorted_indices[:2 * needed_count]  # low confidence means hard samples
                    selected_indices = random.sample(indices_of_hardest_samples, needed_count)
                else:
                    relative_synthetic_confidences = np.array(average_synthetic_confidences) - average_real_confidence
                    sorted_indices = sorted(range(len(relative_synthetic_confidences)),
                                            key=lambda idx: relative_synthetic_confidences[idx])
                    indices_of_most_similar_samples = sorted_indices[:2 * needed_count]
                    selected_indices = random.sample(indices_of_most_similar_samples, needed_count)
            selected_images = torch.stack([class_synthetic_images[i] for i in selected_indices])
            all_synthetic_images.append(selected_images)

        # Convert synthetic images to tensors
        all_synthetic_images = torch.cat(all_synthetic_images)
        original_data_samples, original_labels = self.extract_original_data_and_labels(oversample_targets)

        all_images = torch.cat([original_data_samples, all_synthetic_images])
        all_labels = torch.cat([original_labels, torch.tensor(all_synthetic_labels)])

        return all_images, all_labels

    def holdout_oversample(self, oversample_targets: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform random oversampling to match the desired count using the holdout set."""

        holdout_images = [self.holdout_set[idx][0] for idx in range(len(self.holdout_set))]
        print(f'There is a total of {len(holdout_images)} holdout images.')
        holdout_labels = [self.holdout_set[idx][1] for idx in range(len(self.holdout_set))]
        holdout_per_class = defaultdict(list)
        for i, label in enumerate(holdout_labels):
            if label.item() in oversample_targets.keys():
                holdout_per_class[label.item()].append(holdout_images[i])

        resampled_holdout_images, resampled_holdout_labels = [], []
        for class_id in oversample_targets.keys():
            needed_count = oversample_targets[class_id]
            class_holdout_images = holdout_per_class[class_id]
            print(f'Sampling {needed_count} images from {len(class_holdout_images)} images.')
            selected_indices = random.sample(range(len(class_holdout_images)), needed_count)
            selected_images = torch.stack([class_holdout_images[i] for i in selected_indices])
            resampled_holdout_images.append(selected_images)
            resampled_holdout_labels.extend([class_id for _ in range(needed_count)])

        all_synthetic_images = torch.cat(resampled_holdout_images)
        original_data_samples, original_labels = self.extract_original_data_and_labels(oversample_targets)

        all_images = torch.cat([original_data_samples, all_synthetic_images])
        all_labels = torch.cat([original_labels, torch.tensor(resampled_holdout_labels)])

        return all_images, all_labels

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
            return lambda count, hardness: self.random_oversample(count, hardness)
        elif self.oversampling_strategy == 'SMOTE':
            return lambda count, current_indices: self.SMOTE(count, current_indices)
        elif self.oversampling_strategy == 'rEDM':
            return lambda oversample_targets: self.EDM(oversample_targets, 'random')
        elif self.oversampling_strategy == 'hEDM':
            return lambda oversample_targets: self.EDM(oversample_targets, 'hard')
        elif self.oversampling_strategy == 'aEDM':
            return lambda oversample_targets: self.EDM(oversample_targets, 'average')
        elif self.oversampling_strategy == 'holdout':
            return lambda oversample_targets: self.holdout_oversample(oversample_targets)
        elif self.oversampling_strategy == 'none':
            return None
        else:
            raise ValueError(f"Oversampling strategy {self.oversampling_strategy} is not supported.")

    def resample_data(self, samples_per_class: List[int]):
        """
        Perform resampling to match the desired samples_per_class.
        Uses the selected undersampling and oversampling methods.
        """
        # Select oversampling method
        oversample = self.select_oversampling_method()

        # Organize dataset by class
        class_indices = {i: [] for i in range(self.num_classes)}
        for _, label, idx in self.dataset:
            class_indices[label.item()].append(idx)

        # Perform resampling for each class
        resampled_indices, hard_classes_data, hard_classes_labels, desired_counts, current_counts = [], [], [], [], []
        print(f'After resampling the dataset should have {sum(samples_per_class)} data samples.')

        for class_id, hardnesses_within_class in self.hardness_by_class.items():
            desired_count = samples_per_class[class_id]
            desired_counts.append(desired_count)
            current_indices = class_indices[class_id]
            current_counts.append(len(current_indices))

            if len(current_indices) > desired_count and self.undersampling_strategy != 'none':
                class_retain_indices = self.prune_easy(desired_count, hardnesses_within_class)
                resampled_indices.extend(np.array(current_indices)[class_retain_indices])
            elif len(current_indices) < desired_count and self.oversampling_strategy != 'none':
                # Below if block is necessary because SMOTE generates synthetic samples directly (can't use indices).
                if self.oversampling_strategy == 'SMOTE':
                    original_data, generated_data = oversample(desired_count, current_indices)
                    print(f'Generated {len(generated_data)} data samples via SMOTE for class {class_id}.')
                    hard_classes_data.append(torch.cat([original_data, generated_data]))
                    hard_classes_labels.append(torch.full((desired_count,), class_id))
                elif self.oversampling_strategy == 'random':
                    class_add_indices = oversample(desired_count, hardnesses_within_class)
                    resampled_indices.extend(np.array(current_indices)[class_add_indices])
            else:
                # This part is only used for sanity check tests (comparison with experiment1.py)
                resampled_indices.extend(current_indices)

        if self.oversampling_strategy in ['rEDM', 'hEDM', 'aEDM', 'holdout']:
            oversample_targets = {
                class_id: desired_counts[class_id] - current_counts[class_id]
                for class_id in range(self.num_classes)
                if desired_counts[class_id] > current_counts[class_id]
            }
            hard_classes_data, hard_classes_labels = oversample(oversample_targets)
            print(f'Generated {len(hard_classes_data)} data samples via {self.oversampling_strategy}.')
        elif self.oversampling_strategy == 'SMOTE':
            hard_classes_data = torch.cat(hard_classes_data)
            hard_classes_labels = torch.cat(hard_classes_labels)

        if self.oversampling_strategy in ['SMOTE', 'rEDM', 'hEDM', 'aEDM', 'holdout']:
            print(f'Proceeding with {len(hard_classes_data)} data samples from hard classes (real + synthetic data).')
            original_data_samples = torch.stack([self.dataset[idx][0] for idx in range(len(self.dataset))])
            original_labels = torch.stack([self.dataset[idx][1] for idx in range(len(self.dataset))])

            new_data = torch.cat([original_data_samples, hard_classes_data])
            new_labels = torch.cat([original_labels, hard_classes_labels])
            self.dataset = IndexedDataset(TensorDataset(new_data, new_labels))

            hard_classes_start_idx = original_data_samples.size(0)
            synthetic_indices = list(range(hard_classes_start_idx, new_data.size(0)))
            resampled_indices.extend(synthetic_indices)

        return AugmentedSubset(Subset(self.dataset, resampled_indices))
