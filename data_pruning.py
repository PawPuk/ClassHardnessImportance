import os
import pickle
import random
from typing import List, Union

from diffusers import DDPMPipeline
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import TensorDataset, Subset
import torchvision
from tqdm import tqdm

from config import get_config, ROOT
from data import load_dataset, AugmentedSubset, IndexedDataset


class DataResampling:
    def __init__(self, dataset, num_classes, oversampling_strategy, undersampling_strategy, hardness, high_is_hard):
        self.dataset = dataset
        self.num_classes = num_classes
        self.oversampling_strategy = oversampling_strategy
        self.undersampling_strategy = undersampling_strategy
        self.hardness = hardness
        self.high_is_hard = high_is_hard

    def prune_easy(self, desired_count, hardness_scores):
        sorted_indices = np.argsort(hardness_scores)
        if self.high_is_hard:
            return sorted_indices[-desired_count:]
        else:
            return sorted_indices[:desired_count]

    @staticmethod
    def random_oversample(desired_count, hardness_scores):
        """
        Perform random oversampling to match the desired count.
        """
        additional_indices = random.choices(range(len(hardness_scores)), k=desired_count - len(hardness_scores))
        return list(range(len(hardness_scores))) + additional_indices

    def oversample_easy(self, desired_count, hardness_scores):
        """
        Perform oversampling with a higher chance of duplicating easy samples.
        """
        # Sort indices by ascending hardness (the easiest samples first)
        if self.high_is_hard:
            sorted_indices = np.argsort(hardness_scores)
        else:
            sorted_indices = np.argsort(hardness_scores)[::-1]
        n = len(hardness_scores)

        # Calculate probabilities using the adjusted exponential formula
        alpha_easy = 5
        normalized_hardness = np.linspace(0, 1, n)
        probabilities_sorted = 0.5 + 0.5 * (1 - np.exp(-alpha_easy * (1 - normalized_hardness))) / (
                1 - np.exp(-alpha_easy))

        # Map sorted probabilities back to the original order
        probabilities = np.zeros(n)
        probabilities[sorted_indices] = probabilities_sorted

        # Perform weighted sampling
        additional_indices = random.choices(range(n), weights=probabilities, k=desired_count - n)

        return list(range(n)) + additional_indices

    def oversample_hard(self, desired_count, hardness_scores):
        """
        Perform oversampling with a higher chance of duplicating hard samples.
        """
        # Sort indices by descending hardness (the hardest samples first)
        if self.high_is_hard:
            sorted_indices = np.argsort(hardness_scores)[::-1]
        else:
            sorted_indices = np.argsort(hardness_scores)
        n = len(hardness_scores)

        # Calculate probabilities using the adjusted exponential formula
        alpha_hard = 5
        normalized_hardness = np.linspace(0, 1, n)
        probabilities_sorted = 0.5 + 0.5 * (1 - np.exp(-alpha_hard * (1 - normalized_hardness))) / (
                1 - np.exp(-alpha_hard))

        # Map sorted probabilities back to the original order
        probabilities = np.zeros(n)
        probabilities[sorted_indices] = probabilities_sorted

        # Perform weighted sampling
        additional_indices = random.choices(range(n), weights=probabilities, k=desired_count - n)

        return list(range(n)) + additional_indices

    def SMOTE(self, desired_count, current_indices, k=5):
        """
        Perform oversampling using SMOTE to match the desired count.
        """
        current_n_samples = len(current_indices)

        data = torch.stack([self.dataset[idx][0] for idx in current_indices])
        data_flattened = data.view(current_n_samples, -1)

        neighbors = NearestNeighbors(n_neighbors=k + 1).fit(data_flattened.numpy())
        _, neighbor_indices = neighbors.kneighbors(data_flattened.numpy())

        synthetic_samples = []
        for _ in range(desired_count - current_n_samples):
            idx = torch.randint(0, current_n_samples, (1,)).item()
            neighbor_idx = torch.randint(1, k + 1, (1,)).item()  # Skip the first neighbor (itself)

            sample = data[idx]
            neighbor = data[neighbor_indices[idx][neighbor_idx]]
            alpha = torch.rand(1).item()
            synthetic_sample = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic_sample)

        synthetic_samples = torch.stack(synthetic_samples, dim=0)
        return data, synthetic_samples

    def DDPM(self, desired_count, class_id, current_indices):
        # This only works with CIFAR-10 so we can hardcode the transformation
        data = torch.stack([self.dataset[idx][0] for idx in current_indices])

        ddpm = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to("cuda")
        resnet = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).cuda()
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])

        class_images = []
        while len(class_images) < desired_count:
            for _ in tqdm(range(desired_count // 50)):
                images = ddpm(batch_size=200).images
                batch = torch.stack([transform(img) for img in images]).cuda()
                with torch.no_grad():
                    outputs = resnet(batch)
                    probs = torch.softmax(outputs, dim=1)
                    confidences, labels = torch.max(probs, dim=1)
                    class_synthetic_indices = labels == class_id
                    class_synthetic_images = images[class_synthetic_indices]
                    class_images.extend(class_synthetic_images)
        class_images = torch.stack([transform(img) for img in class_images])
        return data, class_images[:desired_count]

    def select_undersampling_method(self):
        """
        Select the appropriate undersampling method based on the strategy.
        """
        if self.undersampling_strategy == "easy":
            return lambda count, hardness: self.prune_easy(count, hardness)
        elif self.undersampling_strategy == 'none':
            return None
        else:
            raise ValueError(f"Undersampling strategy {self.undersampling_strategy} is not supported.")

    def select_oversampling_method(self):
        """
        Select the appropriate oversampling method based on the strategy.
        """
        if self.oversampling_strategy == "random":
            return lambda count, hardness: self.random_oversample(count, hardness)
        elif self.oversampling_strategy == "easy":
            return lambda count, hardness: self.oversample_easy(count, hardness)
        elif self.oversampling_strategy == "hard":
            return lambda count, hardness: self.oversample_hard(count, hardness)
        elif self.oversampling_strategy == 'SMOTE':
            return lambda count, current_indices: self.SMOTE(count, current_indices)
        elif self.oversampling_strategy == 'DDPM':
            return lambda count, class_id, current_indices: self.DDPM(count, class_id, current_indices)
        elif self.oversampling_strategy == 'none':
            return None
        else:
            raise ValueError(f"Oversampling strategy {self.oversampling_strategy} is not supported.")

    def extract_data_labels(self):
        """
        Extract data and labels from the dataset in a generic way.
        """
        if isinstance(self.dataset, TensorDataset):
            return self.dataset.tensors[0], self.dataset.tensors[1]

        elif hasattr(self.dataset, "data") and hasattr(self.dataset, "targets"):  # Common for datasets like CIFAR10
            data = torch.stack([img for img, _ in self.dataset])
            labels = torch.tensor([label.item() for _, label in self.dataset])
            return data.float(), labels

        elif isinstance(self.dataset, (IndexedDataset, AugmentedSubset)):
            # Extract data and labels, ignoring the index
            data = torch.stack([data for data, _, _ in self.dataset])
            labels = torch.tensor([label.item() for _, label, _ in self.dataset])
            return data.float(), labels

        else:
            raise TypeError(
                "Unsupported dataset type. Ensure the dataset has `tensors`, `data`, and `targets` attributes or "
                "is an instance of `IndexedDataset` or `AugmentedSubset`.")

    def resample_data(self, samples_per_class):
        """
        Perform resampling to match the desired samples_per_class.
        Uses the selected undersampling and oversampling methods.
        """
        # Select resampling methods
        undersample = self.select_undersampling_method()
        oversample = self.select_oversampling_method()

        # Organize dataset by class
        class_indices = {i: [] for i in range(self.num_classes)}
        for idx, (_, label, _) in enumerate(self.dataset):
            class_indices[label.item()].append(idx)

        # Perform resampling for each class
        resampled_indices, hard_classes_data, hard_classes_labels = [], [], []
        print(f'After resampling the dataset should have {sum(samples_per_class.values())} data samples.')

        for class_id, hardnesses_within_class in self.hardness.items():
            desired_count = samples_per_class[class_id]
            current_indices = np.array(class_indices[class_id])

            if len(current_indices) > desired_count and self.undersampling_strategy != 'none':
                class_retain_indices = undersample(desired_count, hardnesses_within_class)
                resampled_indices.extend(current_indices[class_retain_indices])
            elif len(current_indices) < desired_count and self.oversampling_strategy != 'none':
                # Below if block is necessary because SMOTE generates synthetic samples directly (can't use indices).
                if self.oversampling_strategy in ['SMOTE', 'DDPM']:
                    if self.oversampling_strategy == 'SMOTE':
                        original_data, generated_data = oversample(desired_count, current_indices)
                    else:
                        original_data, generated_data = oversample(desired_count, class_id,
                                                                                    current_indices)
                    print(f'Generated {len(generated_data)} data samples via SMOTE for class {class_id}.')
                    hard_classes_data.append(torch.cat([original_data, generated_data], dim=0))
                    hard_classes_labels.append(torch.full((desired_count,), class_id))
                else:
                    class_add_indices = oversample(desired_count, hardnesses_within_class)
                    resampled_indices.extend(current_indices[class_add_indices])
            else:
                resampled_indices.extend(current_indices)

        if self.oversampling_strategy in ['SMOTE', 'DDPM']:
            existing_data, existing_labels = self.extract_data_labels()
            hard_classes_data = torch.cat(hard_classes_data, dim=0)
            hard_classes_labels = torch.cat(hard_classes_labels, dim=0)

            print(f'Proceeding with {len(hard_classes_data)} data samples from hard classes (real + synthetic data).')

            new_data = torch.cat([existing_data, hard_classes_data], dim=0)
            new_labels = torch.cat([existing_labels, hard_classes_labels], dim=0)
            self.dataset = IndexedDataset(TensorDataset(new_data, new_labels))

            hard_classes_start_idx = len(self.dataset) - hard_classes_data.size(0)
            synthetic_indices = list(range(hard_classes_start_idx, len(self.dataset)))
            resampled_indices.extend(synthetic_indices)

        return AugmentedSubset(Subset(self.dataset, resampled_indices))


class DataPruning:
    def __init__(self, instance_hardness: List[List[Union[int, float]]], prune_percentage: int, dataset_name: str,
                 high_is_hard: bool):
        """
        Initialize the DataPruning class.

        :param instance_hardness: List of lists of instance-level hardness scores.
        :param prune_percentage: Percentage of the data to prune (default is 50%).
        :param dataset_name: Name of the dataset we are working on (used for saving).
        """
        # Compute the average instance-level hardness for each sample across all models
        self.instance_hardness = np.mean(np.array(instance_hardness), axis=0)
        self.prune_percentage = prune_percentage / 100
        self.dataset_name = dataset_name
        self.high_is_hard = high_is_hard

        self.fig_save_dir = os.path.join(ROOT, 'Figures/')
        self.res_save_dir = os.path.join(ROOT, 'Results/')
        self.num_classes = get_config(dataset_name)['num_classes']
        self.class_level_sample_counts = {}

    def get_unpruned_indices(self, hardness_scores: NDArray[Union[int, float]], retain_count: int) -> NDArray[np.int_]:
        sorted_indices = np.argsort(hardness_scores)

        if self.high_is_hard:
            return sorted_indices[-retain_count:]
        else:
            return sorted_indices[:retain_count]

    def plot_class_level_sample_distribution(self, remaining_indices: List[int], pruning_key: str, labels: NDArray[int]):
        os.makedirs(self.fig_save_dir, exist_ok=True)
        os.makedirs(self.res_save_dir, exist_ok=True)

        # Count the number of remaining samples for each class
        remaining_labels = labels[remaining_indices]
        unique_classes, class_counts = np.unique(remaining_labels, return_counts=True)

        # Create a dictionary for current pruning type if it doesn't exist
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
        plt.savefig(os.path.join(self.fig_save_dir, "class_level_sample_distribution.pdf"))
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
        plt.savefig(os.path.join(self.fig_save_dir, "sorted_class_level_sample_distribution.pdf"))
        plt.close()

    def dataset_level_pruning(self, labels: NDArray[int]) -> list[int]:
        """
        Remove the specified percentage of samples with the lowest instance-level hardness.
        This method returns the indices of the remaining data after pruning.

        :return: List of indices of the remaining data samples after pruning.
        """
        retain_count = int((1 - self.prune_percentage) * len(self.instance_hardness))
        remaining_indices = self.get_unpruned_indices(self.instance_hardness, retain_count)

        self.fig_save_dir = os.path.join(self.fig_save_dir, 'dlp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.res_save_dir = os.path.join(self.res_save_dir, 'dlp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.plot_class_level_sample_distribution(remaining_indices.tolist(), 'dlp', labels)

        return remaining_indices.tolist()

    def fixed_class_level_pruning(self, labels: NDArray[int]) -> list[int]:
        """
        Remove the specified percentage of samples from each class.
        Ensures that the class distribution remains balanced.

        :return: List of indices of the remaining data samples after class-level pruning.
        """
        remaining_indices = []
        class_level_hardness = {class_id: np.array([]) for class_id in range(self.num_classes)}

        _, training_dataset, _, _ = load_dataset(self.dataset_name, False, False, True)
        for i, (_, label, _) in enumerate(training_dataset):
            np.append(class_level_hardness[label], self.instance_hardness[i])

        for class_id, class_scores in class_level_hardness.items():
            retain_count = int((1 - self.prune_percentage) * len(class_scores))
            class_remaining_indices = self.get_unpruned_indices(class_scores, retain_count)
            global_indices = np.where(labels == class_id)[0]
            remaining_indices.extend(global_indices[class_remaining_indices])

        self.fig_save_dir = os.path.join(self.fig_save_dir, 'fclp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.res_save_dir = os.path.join(self.res_save_dir, 'fclp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.plot_class_level_sample_distribution(remaining_indices, 'fclp', labels)

        return remaining_indices
