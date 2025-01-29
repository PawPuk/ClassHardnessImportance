import os
import pickle
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

from neural_networks import ResNet18LowRes
from utils import get_config, AugmentedSubset, IndexedDataset


class DataResampling:
    def __init__(self, dataset, num_classes, oversampling_strategy, undersampling_strategy, hardness, dataset_name):
        """
        Initialize with the dataset, number of classes, and resampling strategies.
        """
        self.dataset = dataset
        self.num_classes = num_classes
        self.oversampling_strategy = oversampling_strategy
        self.undersampling_strategy = undersampling_strategy
        self.hardness = hardness
        self.fig_save_dir = 'Figures/'
        self.model_save_dir = f'Models/none/{dataset_name}/'

    def plot_probability_distribution(self, probabilities, method_name, class_id):
        """
        Plot and save both unsorted and sorted probability distributions for oversampling.
        """
        os.makedirs(self.fig_save_dir, exist_ok=True)

        # Plot the unsorted probabilities
        plt.figure()
        plt.plot(probabilities, marker='o', linestyle='-', alpha=0.75)
        plt.title(f'Probability Distribution ({method_name.capitalize()} Oversampling, Class {class_id}) - Unsorted')
        plt.xlabel('Samples')
        plt.ylabel('Probability')
        plt.grid(True)

        # Save the unsorted plot
        plot_filename_unsorted = os.path.join(
            self.fig_save_dir,
            f'{method_name}_oversampling_class_{class_id}_probabilities_unsorted.png'
        )
        plt.savefig(plot_filename_unsorted)
        plt.close()

        # Plot the sorted probabilities
        sorted_probabilities = np.sort(probabilities)[::-1]

        plt.figure()
        plt.plot(sorted_probabilities, marker='o', linestyle='-', alpha=0.75)
        plt.title(f'Sorted Probability Distribution ({method_name.capitalize()} Oversampling, Class {class_id})')
        plt.xlabel('Samples (sorted)')
        plt.ylabel('Probability')
        plt.grid(True)

        # Save the sorted plot
        plot_filename_sorted = os.path.join(
            self.fig_save_dir,
            f'{method_name}_oversampling_class_{class_id}_probabilities_sorted.png'
        )
        plt.savefig(plot_filename_sorted)
        plt.close()

    @staticmethod
    def random_undersample(desired_count, hardness_scores):
        """
        Perform random undersampling to match the desired count.
        """
        return random.sample(range(len(hardness_scores)), desired_count)

    @staticmethod
    def prune_easy(desired_count, hardness_scores):
        """
        Perform undersampling by pruning the easiest samples based on hardness scores.
        """
        sorted_indices = np.argsort(hardness_scores)
        return sorted_indices[-desired_count:]

    @staticmethod
    def prune_hard(desired_count, hardness_scores):
        """
        Perform undersampling by pruning the hardest samples based on hardness scores.
        """
        sorted_indices = np.argsort(hardness_scores)
        return sorted_indices[:desired_count]

    @staticmethod
    def prune_extreme(desired_count, hardness_scores):
        """
        Perform undersampling by firstly pruning 1% of the hardest samples and then pruning the easy samples.
        """
        sorted_indices = np.argsort(hardness_scores)
        one_percent_count = max(1, desired_count // 100)
        ninety_nine_percent_count = desired_count - one_percent_count

        hardest_indices = sorted_indices[-one_percent_count:]
        easiest_indices = sorted_indices[:ninety_nine_percent_count]
        return np.concatenate([easiest_indices, hardest_indices])

    @staticmethod
    def random_oversample(desired_count, hardness_scores):
        """
        Perform random oversampling to match the desired count.
        """
        additional_indices = random.choices(range(len(hardness_scores)), k=desired_count - len(hardness_scores))
        return list(range(len(hardness_scores))) + additional_indices

    @staticmethod
    def plot_and_save_synthetic_samples(synthetic_data, filename):
        """
        Create a 4x15 plot of 60 synthetic samples and save it to a file.
        """
        fig, axes = plt.subplots(4, 15, figsize=(15, 8))
        axes = axes.flatten()

        CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
        CIFAR10_STD = torch.tensor([0.247, 0.243, 0.261])

        for i in range(60):
            unnormalized_image = synthetic_data[i] * CIFAR10_STD[:, None, None] + CIFAR10_MEAN[:, None, None]
            axes[i].imshow(unnormalized_image.permute(1, 2, 0).numpy())  # Convert from CxHxW to HxWxC for image display
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def SMOTE(self, desired_count, hardness_scores, current_indices, hardness_stats, k=5):
        """
        Perform oversampling using SMOTE to match the desired count.
        """
        current_n_samples = len(current_indices)

        data = torch.stack([self.dataset[idx][0] for idx in current_indices])
        labels = torch.tensor([self.dataset[idx][1] for idx in current_indices])
        data_flattened = data.view(current_n_samples, -1)

        neighbors = NearestNeighbors(n_neighbors=k + 1).fit(data_flattened.numpy())
        _, neighbor_indices = neighbors.kneighbors(data_flattened.numpy())

        synthetic_samples = []
        print('-'*20)
        print(desired_count, current_n_samples, desired_count - current_n_samples)
        print()
        for _ in range(desired_count - current_n_samples):
            idx = torch.randint(0, current_n_samples, (1,)).item()
            neighbor_idx = torch.randint(1, k + 1, (1,)).item()  # Skip the first neighbor (itself)

            sample = data[idx]
            sample_hardness = hardness_scores[idx]
            neighbor = data[neighbor_indices[idx][neighbor_idx]]
            neighbor_hardness = hardness_scores[neighbor_indices[idx][neighbor_idx]]

            hardness_stats['avg_pair_hardness'].append(sample_hardness)
            hardness_stats['avg_pair_hardness'].append(neighbor_hardness)
            hardness_stats['avg_hardness_diff_within_pair'].append(abs(sample_hardness - neighbor_hardness))

            # Interpolate between the sample and its neighbor
            alpha = torch.rand(1).item()
            synthetic_sample = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic_sample)

        synthetic_samples = torch.stack(synthetic_samples, dim=0)
        # self.plot_and_save_synthetic_samples(synthetic_samples, f'Figures/synthetic_samples_class_{class_id}.png')
        return data, labels, synthetic_samples

    def oversample_easy(self, desired_count, hardness_scores, class_id):
        """
        Perform oversampling with a higher chance of duplicating easy samples.
        """
        # Sort indices by ascending hardness (the easiest samples first)
        sorted_indices = np.argsort(hardness_scores)
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

        # Plot and save the probability distribution
        self.plot_probability_distribution(probabilities, 'easy', class_id)

        return list(range(n)) + additional_indices

    def oversample_hard(self, desired_count, hardness_scores, class_id):
        """
        Perform oversampling with a higher chance of duplicating hard samples.
        """
        # Sort indices by descending hardness (the hardest samples first)
        sorted_indices = np.argsort(hardness_scores)[::-1]
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

        # Plot and save the probability distribution
        self.plot_probability_distribution(probabilities, 'hard', class_id)

        return list(range(n)) + additional_indices

    def select_undersampling_method(self):
        """
        Select the appropriate undersampling method based on the strategy.
        """
        if self.undersampling_strategy == "random":
            return lambda count, hardness: self.random_undersample(count, hardness)
        elif self.undersampling_strategy == "easy":
            return lambda count, hardness: self.prune_easy(count, hardness)
        elif self.undersampling_strategy == 'hard':
            return lambda count, hardness: self.prune_hard(count, hardness)
        elif self.undersampling_strategy == 'extreme':
            return lambda count, hardness: self.prune_extreme(count, hardness)
        else:
            raise ValueError(f"Undersampling strategy {self.undersampling_strategy} is not supported.")

    def select_oversampling_method(self):
        """
        Select the appropriate oversampling method based on the strategy.
        """
        if self.oversampling_strategy == "random":
            return lambda count, hardness, class_id: self.random_oversample(count, hardness)
        elif self.oversampling_strategy == "easy":
            return lambda count, hardness, class_id: self.oversample_easy(count, hardness, class_id)
        elif self.oversampling_strategy == "hard":
            return lambda count, hardness, class_id: self.oversample_hard(count, hardness, class_id)
        elif self.oversampling_strategy == 'SMOTE':
            return lambda count, hardness, current_indices, stats: self.SMOTE(count, hardness, current_indices, stats)
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
            labels = torch.tensor([label for _, label in self.dataset])
            return data.float(), labels

        elif isinstance(self.dataset, (IndexedDataset, AugmentedSubset)):
            # Extract data and labels, ignoring the index
            data = torch.stack([data for data, _, _ in self.dataset])
            labels = torch.tensor([label for _, label, _ in self.dataset])
            return data.float(), labels

        else:
            raise TypeError(
                "Unsupported dataset type. Ensure the dataset has `tensors`, `data`, and `targets` attributes or "
                "is an instance of `IndexedDataset` or `AugmentedSubset`.")

    def load_model_states(self):
        if os.path.exists(self.model_save_dir):
            model_states = []
            for file in os.listdir(self.model_save_dir):
                if "_epoch_20.pth" in file:
                    model_path = os.path.join(self.model_save_dir, file)
                    model_state = torch.load(model_path)
                    model_states.append(model_state)
        else:
            raise Exception
        return model_states

    def estimate_hardness(self, data, labels, all_stats):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        synthetic_dataset = TensorDataset(data, labels)
        synthetic_dataloader = DataLoader(synthetic_dataset, batch_size=1000, shuffle=False)
        model_states = self.load_model_states()

        ensemble_stats = []
        for model_state in model_states:
            model = ResNet18LowRes(self.num_classes)
            model.load_state_dict(model_state)
            model = model.to(device)
            model.eval()
            model_stats = []
            with torch.no_grad():
                for images, labels in synthetic_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                    softmax_outputs = F.softmax(outputs, dim=1)
                    one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()
                    l2_errors = torch.norm(softmax_outputs - one_hot_labels, dim=1)
                    model_stats.extend(l2_errors.cpu().numpy())
            ensemble_stats.append(model_stats)
        averaged_stats = [sum(model_stats) / len(model_stats) for model_stats in zip(*ensemble_stats)]
        all_stats['avg_synthetic_data_hardness'] = averaged_stats

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
            class_indices[label].append(idx)

        # Perform resampling for each class
        resampled_indices, synthetic_data, synthetic_labels = [], [], []
        hardness_stats = {'avg_pair_hardness': [], 'avg_hardness_diff_within_pair': [],
                          'avg_synthetic_data_hardness': [], 'avg_respective_synthetic_data_hardness': []}

        for class_id, hardnesses_within_class in self.hardness.items():
            desired_count = samples_per_class[class_id]
            current_indices = np.array(class_indices[class_id])

            if len(current_indices) > desired_count:
                class_retain_indices = undersample(desired_count, hardnesses_within_class)
                resampled_indices.extend(current_indices[class_retain_indices])
            elif len(current_indices) < desired_count:
                if self.oversampling_strategy in ['SMOTE', 'BSMOTE']:
                    # This if block is necessary because SMOTE generates synthetic samples directly (can't use indices).
                    original_data, original_labels, generated_data = oversample(desired_count, hardnesses_within_class,
                                                                                current_indices, hardness_stats)
                    generated_labels = torch.full((generated_data.size(0),), class_id)
                    print(f'Generated {len(generated_data)} data samples via SMOTE.')
                    if len(generated_data) + len(current_indices) != desired_count:
                        print(len(generated_data), len(current_indices), desired_count)
                        raise Exception
                    synthetic_data.append(torch.cat([original_data, generated_data], dim=0))
                    synthetic_labels.append(torch.cat([original_labels, generated_labels], dim=0))
                else:
                    class_add_indices = oversample(desired_count, hardnesses_within_class, class_id)
                    resampled_indices.extend(current_indices[class_add_indices])
            else:
                resampled_indices.extend(current_indices)

        if synthetic_data:
            existing_data, existing_labels = self.extract_data_labels()
            synthetic_data = torch.cat(synthetic_data, dim=0)
            synthetic_labels = torch.cat(synthetic_labels, dim=0)

            # self.estimate_hardness(synthetic_data, synthetic_labels, hardness_stats)
            print(f'Generated {len(synthetic_data)} synthetic data samples.')
            print(hardness_stats)
            for key, item in hardness_stats.items():
                print(f'Key {key} has {len(item)} elements.')

            new_data = torch.cat([existing_data, synthetic_data], dim=0)
            new_labels = torch.cat([existing_labels, synthetic_labels], dim=0)
            self.dataset = IndexedDataset(TensorDataset(new_data, new_labels))

            synthetic_start_idx = len(self.dataset) - synthetic_data.size(0)
            synthetic_indices = list(range(synthetic_start_idx, len(self.dataset)))
            resampled_indices.extend(synthetic_indices)

        # Create the resampled dataset
        return Subset(self.dataset, resampled_indices)


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
        self.labels = np.array(labels)
        self.prune_percentage = prune_percentage / 100
        self.dataset_name = dataset_name
        self.protect_prototypes = protect_prototypes
        self.fig_save_dir = 'Figures/'
        self.res_save_dir = 'Results/'
        self.num_classes = get_config(dataset_name)['num_classes']

        # Compute the average class-level hardness for each sample in specific class across all models
        if self.protect_prototypes:
            self.class_hardness = {}
            for class_id, class_scores in class_hardness.items():
                # Prototypes should not be taken under consideration when measuring hardness, as they are not pruned.
                one_percent = int(0.01 * len(class_scores))
                self.class_hardness[class_id] = np.mean(np.array(class_scores)[one_percent:], axis=1)
        else:
            self.class_hardness = {
                class_id: np.mean(np.array(class_scores), axis=1)
                for class_id, class_scores in class_hardness.items()
            }

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
            hardest_samples_count = retain_count - one_percent
            easiest_indices = sorted_indices[:one_percent]
            hardest_indices = sorted_indices[-hardest_samples_count:]
            return np.concatenate((easiest_indices, hardest_indices))
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
        with open(os.path.join(self.res_save_dir, f"{['unprotected', 'protected'][self.protect_prototypes]}"
                                                    f"_class_level_sample_counts.pkl"), "wb") as file:
            pickle.dump(self.class_level_sample_counts, file)

        # Plot the original class distribution
        plt.figure()
        plt.bar(unique_classes, class_counts)
        plt.ylabel('Number of Remaining Samples')
        plt.title(f'Class-level Distribution of Remaining Samples After {pruning_key.upper()} Pruning')
        plt.xticks([])
        plt.savefig(os.path.join(self.fig_save_dir, f"{['unprotected', 'protected'][self.protect_prototypes]}"
                                                    f"_class_level_sample_distribution.pdf"))
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
        plt.savefig(os.path.join(self.fig_save_dir, f"{['unprotected', 'protected'][self.protect_prototypes]}"
                                                    f"_sorted_class_level_sample_distribution.pdf"))
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
