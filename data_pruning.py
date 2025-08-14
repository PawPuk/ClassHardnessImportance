from collections import defaultdict
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

from config import get_config, ROOT
from data import load_dataset, AugmentedSubset, IndexedDataset
from neural_networks import ResNet18LowRes
from utils import load_results


class DataResampling:
    def __init__(self, dataset, num_classes, oversampling_strategy, undersampling_strategy, hardness, high_is_hard,
                 dataset_name, num_models_for_hardness, mean, std):
        self.dataset = dataset
        self.num_classes = num_classes
        self.oversampling_strategy = oversampling_strategy
        self.undersampling_strategy = undersampling_strategy
        self.hardness = hardness
        self.high_is_hard = high_is_hard
        self.dataset_name = dataset_name
        self.num_models_for_hardness = num_models_for_hardness
        self.mean = mean
        self.std = std

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

    def extract_original_data_and_labels(self, oversample_targets):
        original_data, original_labels = [], []
        for idx in range(len(self.dataset)):
            image, label, _ = self.dataset[idx]
            if label.item() in oversample_targets:
                original_data.append(image)
                original_labels.append(label)
        return original_data, original_labels

    def DDPM(self, oversample_targets):
        to_tensor = torchvision.transforms.ToTensor()
        original_data, original_labels = self.extract_original_data_and_labels(oversample_targets)
        if not oversample_targets:
            return original_data, original_labels

        ddpm = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to("cuda")
        resnet = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).cuda()
        normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        synthetic_per_class = defaultdict(list)

        while any(len(synthetic_per_class[c]) < oversample_targets[c] for c in oversample_targets):
            images = ddpm(batch_size=1000).images
            batch = torch.stack([normalize(to_tensor(img)) for img in images]).cuda()
            with torch.no_grad():
                outputs = resnet(batch)
                probs = torch.softmax(outputs, dim=1)
                _, predicted_labels = torch.max(probs, dim=1)
            for i, label in enumerate(predicted_labels):
                label = label.item()
                if label in oversample_targets and len(synthetic_per_class[label]) < oversample_targets[label]:
                    synthetic_per_class[label].append(to_tensor(images[i]))

            samples_to_generate = sum(
                oversample_targets[class_id] - len(synthetic_per_class.get(class_id, []))
                for class_id in oversample_targets
            )
            print(f'\n\n\nStill need to generate {samples_to_generate} data samples.\n\n\n')

        all_synthetic_images, all_synthetic_labels = [], []
        for class_id in oversample_targets:
            all_synthetic_images.extend(synthetic_per_class[class_id][:oversample_targets[class_id]])
            all_synthetic_labels.extend([class_id] * oversample_targets[class_id])

        all_images = torch.stack(original_data + all_synthetic_images)  # Shape: [N, 3, 32, 32]
        all_labels = torch.tensor(original_labels + all_synthetic_labels)  # Shape: [N]

        return all_images, all_labels

    def load_model_states(self):
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

    def compute_confidences(self, model_states, images, class_id, batch_size=1024):
        """For a given class_id, compute average confidence across models for each synthetic image."""
        num_samples, avg_confidences = len(images), []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_images = images[batch_start:batch_end]

            to_tensor = torchvision.transforms.ToTensor()
            normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
            img_tensor = [to_tensor(img) if not isinstance(img, torch.Tensor) else img for img in batch_images]
            normalized_images = [normalize(img) for img in img_tensor]
            batch_normalized_images = torch.stack(normalized_images).cuda()  # Shape: [B, 3, 32, 32]

            # For each model, compute confidence
            batch_confidences = torch.zeros(batch_normalized_images.size(0), device='cuda')
            for model_state in model_states:
                model = ResNet18LowRes(self.num_classes)
                model.load_state_dict(model_state)
                model = model.cuda()
                model.eval()
                with torch.no_grad():
                    logits = model(batch_normalized_images)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    conf = probs[:, class_id]  # confidence for true class
                    batch_confidences += conf  # accumulate per model

            batch_confidences /= len(model_states)  # average confidence across models
            avg_confidences.extend(batch_confidences.cpu().tolist())

        return avg_confidences

    def EDM(self, oversample_targets, strategy):
        # TODO: Mention downloading images in documentation

        # Load the synthetic data generated by EDM
        synthetic_data = np.load(os.path.join(ROOT, 'GeneratedImages', f'{self.dataset_name}.npz'))
        synthetic_images = synthetic_data[synthetic_data.files[0]]
        synthetic_labels = synthetic_data[synthetic_data.files[1]]

        # Save the original data (we will later add EDM data to it to produce resampled dataset)
        original_data, original_labels = self.extract_original_data_and_labels(oversample_targets)
        if not oversample_targets:
            return original_data, original_labels

        # Only continue with samples from the classes that we want to oversample
        synthetic_per_class, real_per_class = defaultdict(list), defaultdict(list)
        for i, label in enumerate(synthetic_labels):
            if label in oversample_targets:
                synthetic_per_class[label].append(synthetic_images[i])
        for image, label, _ in self.dataset:
            if label.item() in oversample_targets:
                real_per_class[label].append(image)

        # Iterate through classes that we want to oversample
        all_synthetic_images, all_synthetic_labels = [], []
        for class_id in oversample_targets:
            needed_count = oversample_targets[class_id]
            all_synthetic_labels.extend([class_id for _ in range(needed_count)])
            class_synthetic_images = synthetic_per_class[class_id]
            class_real_images = real_per_class[class_id]
            if strategy == 'random':
                selected_images = random.sample(class_synthetic_images, needed_count)
            else:
                model_states = self.load_model_states()
                average_synthetic_confidences = self.compute_confidences(model_states, class_synthetic_images, class_id)
                average_real_confidence = np.mean(self.compute_confidences(model_states, class_real_images, class_id))
                if strategy == 'hard':
                    sorted_indices = sorted(range(len(average_synthetic_confidences)),
                                            key=lambda idx: average_synthetic_confidences[idx])
                    indices_of_hardest_samples = sorted_indices[:2 * needed_count]
                    selected_indices = random.sample(indices_of_hardest_samples, needed_count)
                else:
                    relative_synthetic_confidences = average_synthetic_confidences - average_real_confidence
                    sorted_indices = sorted(range(len(relative_synthetic_confidences)),
                                            key=lambda  idx: relative_synthetic_confidences[idx])
                    indices_of_most_similar_samples = sorted_indices[:2 * needed_count]
                    selected_indices = random.sample(indices_of_most_similar_samples, needed_count)
                selected_images = [class_synthetic_images[i] for i in selected_indices]
            all_synthetic_images.extend(selected_images)

        # Convert synthetic images to tensors to match the DDPM output format
        to_tensor = torchvision.transforms.ToTensor()
        all_synthetic_tensors = [to_tensor(img) if not isinstance(img, torch.Tensor) else img
                                 for img in all_synthetic_images]
        all_images = torch.stack(original_data + all_synthetic_tensors)  # Shape: [N, 3, 32, 32]
        all_labels = torch.tensor(original_labels + all_synthetic_labels)  # Shape: [N]

        return all_images, all_labels

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
            return lambda oversample_targets: self.DDPM(oversample_targets)
        elif self.oversampling_strategy == 'rEDM':
            return lambda oversample_targets: self.EDM(oversample_targets, 'random')
        elif self.oversampling_strategy == 'hEDM':
            return lambda oversample_targets: self.EDM(oversample_targets, 'hard')
        elif self.oversampling_strategy == 'aEDM':
            return lambda oversample_targets: self.EDM(oversample_targets, 'average')
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
        resampled_indices, hard_classes_data, hard_classes_labels, desired_counts, current_counts = [], [], [], [], []
        print(f'After resampling the dataset should have {sum(samples_per_class.values())} data samples.')

        for class_id, hardnesses_within_class in self.hardness.items():
            desired_count = samples_per_class[class_id]
            desired_counts.append(desired_count)
            current_indices = np.array(class_indices[class_id])
            current_counts.append(len(current_indices))

            if len(current_indices) > desired_count and self.undersampling_strategy != 'none':
                class_retain_indices = undersample(desired_count, hardnesses_within_class)
                resampled_indices.extend(current_indices[class_retain_indices])
            elif len(current_indices) < desired_count and self.oversampling_strategy != 'none':
                # Below if block is necessary because SMOTE generates synthetic samples directly (can't use indices).
                if self.oversampling_strategy in ['SMOTE']:
                    original_data, generated_data = oversample(desired_count, current_indices)
                    print(f'Generated {len(generated_data)} data samples via SMOTE for class {class_id}.')
                    hard_classes_data.append(torch.cat([original_data, generated_data], dim=0))
                    hard_classes_labels.append(torch.full((desired_count,), class_id))
                elif self.oversampling_strategy in ['easy', 'hard', 'random']:
                    class_add_indices = oversample(desired_count, hardnesses_within_class)
                    resampled_indices.extend(current_indices[class_add_indices])
            else:
                resampled_indices.extend(current_indices)
        if self.oversampling_strategy in ['DDPM', 'rEDM', 'hEDM', 'aEDM']:
            oversample_targets = {
                class_id: desired_counts[class_id] - current_counts[class_id]
                for class_id in range(self.num_classes)
                if desired_counts[class_id] > current_counts[class_id]
            }
            hard_classes_data, hard_classes_labels = oversample(oversample_targets)
            print(f'Generated {len(hard_classes_data)} data samples via {self.oversampling_strategy}.')
        elif self.oversampling_strategy == 'SMOTE':
            hard_classes_data = torch.cat(hard_classes_data, dim=0)
            hard_classes_labels = torch.cat(hard_classes_labels, dim=0)

        if self.oversampling_strategy in ['SMOTE', 'DDPM', 'rEDM', 'hEDM', 'aEDM']:
            print(f'Proceeding with {len(hard_classes_data)} data samples from hard classes (real + synthetic data).')
            existing_data, existing_labels = self.extract_data_labels()

            new_data = torch.cat([existing_data, hard_classes_data], dim=0)
            new_labels = torch.cat([existing_labels, hard_classes_labels], dim=0)
            self.dataset = IndexedDataset(TensorDataset(new_data, new_labels))

            hard_classes_start_idx = len(self.dataset) - hard_classes_data.size(0)
            synthetic_indices = list(range(hard_classes_start_idx, len(self.dataset)))
            resampled_indices.extend(synthetic_indices)

        return AugmentedSubset(Subset(self.dataset, resampled_indices))


class DataPruning:
    def __init__(self, instance_hardness: Union[NDArray, None], prune_percentage: int,
                 dataset_name: str, high_is_hard: Union[bool, None], imbalance_ratio: Union[List[int], None]):
        """
        Initialize the DataPruning class.

        :param instance_hardness: List of lists of instance-level hardness scores.
        :param prune_percentage: Percentage of the data to prune (default is 50%).
        :param dataset_name: Name of the dataset we are working on (used for saving).
        """
        # Compute the average instance-level hardness for each sample across all models
        if instance_hardness is not None:
            self.instance_hardness = np.mean(np.array(instance_hardness), axis=0)
        self.prune_percentage = prune_percentage / 100
        self.dataset_name = dataset_name
        self.high_is_hard = high_is_hard
        self.imbalance_ratio = imbalance_ratio

        config = get_config(dataset_name)
        self.num_classes = config['num_classes']
        self.num_samples_per_class = config['num_training_samples']

        self.fig_save_dir = os.path.join(ROOT, 'Figures/')
        self.res_save_dir = os.path.join(ROOT, 'Results/')

    def get_unpruned_indices(self, hardness_scores: Union[NDArray[Union[int, float]], None], retain_count: int,
                             current_number_of_samples: int) -> NDArray[np.int_]:
        if hardness_scores is None:
            return np.random.choice(current_number_of_samples, retain_count, replace=False)
        else:
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

        path = os.path.join(self.res_save_dir, 'class_level_sample_counts.pkl')
        if os.path.exists(path) and os.path.getsize(path) > 0:
            class_level_sample_counts_after_pruning = load_results(path)
        else:
            class_level_sample_counts_after_pruning = {}
        if pruning_key not in class_level_sample_counts_after_pruning:
            class_level_sample_counts_after_pruning[pruning_key] = {}

        # Store the distribution of samples after pruning
        class_level_sample_counts_after_pruning[pruning_key][int(self.prune_percentage * 100)] = [
            class_counts[unique_classes.tolist().index(cls)] if cls in unique_classes else 0
            for cls in range(self.num_classes)
        ]
        with open(os.path.join(self.res_save_dir, "class_level_sample_counts.pkl"), "wb") as file:
            pickle.dump(class_level_sample_counts_after_pruning, file)

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
        if self.imbalance_ratio is None:
            retain_count = int((1 - self.prune_percentage) * sum(self.num_samples_per_class))
            remaining_indices = self.get_unpruned_indices(self.instance_hardness, retain_count,
                                                          sum(self.num_samples_per_class))
        else:
            remaining_indices = []
            for class_id in range(self.num_classes):
                class_remaining_indices = self.get_unpruned_indices(None, self.imbalance_ratio[class_id],
                                                                    self.num_samples_per_class[class_id])
                global_indices = np.where(labels == class_id)[0]
                remaining_indices.extend(global_indices[class_remaining_indices])
            remaining_indices = np.array(remaining_indices)

        self.fig_save_dir = os.path.join(self.fig_save_dir, 'random_dlp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.res_save_dir = os.path.join(self.res_save_dir, 'random_dlp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.plot_class_level_sample_distribution(remaining_indices.tolist(), 'dlp', labels)

        return remaining_indices.tolist()

    def class_level_pruning(self, labels: NDArray[int]) -> list[int]:
        """
        Remove the specified percentage of samples from each class.
        Ensures that the class distribution remains balanced.

        :return: List of indices of the remaining data samples after class-level pruning.
        """
        remaining_indices = []
        class_level_hardness = {class_id: np.array([]) for class_id in range(self.num_classes)}

        if self.imbalance_ratio is None:
            _, training_dataset, _, _ = load_dataset(self.dataset_name, False, False, True)
            for i, (_, label, _) in enumerate(training_dataset):
                class_level_hardness[label] = np.append(class_level_hardness[label], self.instance_hardness[i])

        for class_id in range(self.num_classes):
            class_scores = None if self.imbalance_ratio is not None else class_level_hardness[class_id]
            retain_count = int((1 - self.prune_percentage) * self.num_samples_per_class[class_id])
            class_remaining_indices = self.get_unpruned_indices(class_scores, retain_count,
                                                                self.num_samples_per_class[class_id])
            global_indices = np.where(labels == class_id)[0]
            remaining_indices.extend(global_indices[class_remaining_indices])

        self.fig_save_dir = os.path.join(self.fig_save_dir, 'random_clp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.res_save_dir = os.path.join(self.res_save_dir, 'random_clp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.plot_class_level_sample_distribution(remaining_indices, 'clp', labels)

        return remaining_indices
