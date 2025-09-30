"""Core module for pruning the dataset that returns the pruned subdataset."""

import os
import pickle
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Subset
import tqdm

from config import DEVICE, get_config, ROOT
from data import AugmentedSubset, get_dataloader, IndexedDataset
from data_resampling import DataResampling
from neural_networks import ResNet18LowRes
from train_ensemble import ModelTrainer
from utils import load_results


class DataPruning:
    """Class that contains all the methods required for producing the pruned subdatasets. It outputs the pruned
    subdataset."""
    def __init__(self, prune_percentage: int, dataset_name: str, imbalance_ratio: List[int]):
        """
        Initialize the DataPruning class.

        :param prune_percentage: Percentage of the data to prune.
        :param dataset_name: Name of the dataset we are working on (used for saving).
        :param imbalance_ratio: List of integers specifying the number of samples to be left in each class after
        hardness-based resampling - for resampling_pruned_subdataset().
        """
        self.prune_percentage = prune_percentage / 100
        self.dataset_name = dataset_name
        self.imbalance_ratio = imbalance_ratio

        config = get_config(dataset_name)
        self.num_classes = config['num_classes']
        self.num_samples_per_class = config['num_training_samples']
        self.num_models_for_hardness = config['num_models_for_hardness']
        self.num_models_per_dataset = config['num_models_per_dataset']
        self.mean = config['mean']
        self.std = config['std']

        self.fig_save_dir = os.path.join(ROOT, 'Figures/')
        self.res_save_dir = os.path.join(ROOT, 'Results/')

    def plot_class_level_sample_distribution(self, subdataset_indices: List[int], pruning_key: str, labels: List[int]):
        """Visualizes the per-class sample count after pruning, and saves the class_level_sample_counts.pkl"""
        os.makedirs(self.fig_save_dir, exist_ok=True)
        os.makedirs(self.res_save_dir, exist_ok=True)

        # Count the number of remaining samples for each class
        remaining_labels = np.array(labels)[subdataset_indices]
        unique_classes, class_counts = np.unique(remaining_labels, return_counts=True)

        path = os.path.join(self.res_save_dir, 'class_level_sample_counts.pkl')
        if os.path.exists(path) and os.path.getsize(path) > 0:
            class_level_sample_counts_after_pruning = load_results(path)
        else:
            class_level_sample_counts_after_pruning = {}
        if pruning_key not in class_level_sample_counts_after_pruning:
            class_level_sample_counts_after_pruning[pruning_key] = {}

        # Store the distribution of samples after pruning
        class_level_sample_counts_after_pruning[pruning_key][int(round(self.prune_percentage * 100))] = [
            class_counts[unique_classes.tolist().index(cls)] if cls in unique_classes else 0
            for cls in range(self.num_classes)
        ]
        with open(os.path.join(self.res_save_dir, "class_level_sample_counts.pkl"), "wb") as file:
            # noinspection PyTypeChecker
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

    def class_level_pruning(self, labels: List[int], training_dataset: Union[AugmentedSubset, IndexedDataset]
                            ) -> Tuple[AugmentedSubset, AugmentedSubset]:
        """
        Remove the specified percentage of samples from each class. Ensures that the class distribution remains
        balanced.

        :param labels: List of labels for each data sample in the original dataset.
        :param training_dataset: Original training dataset

        :return: Balanced subdataset, containing the remaining samples, and the pruned_dataset containing the pruned
        samples.
        """
        subdataset_indices = []

        for class_id in range(self.num_classes):
            retain_count = int((1 - self.prune_percentage) * self.num_samples_per_class[class_id])
            class_remaining_indices = np.random.choice(self.num_samples_per_class[class_id], retain_count,
                                                       replace=False)
            global_indices = np.where(np.array(labels) == class_id)[0]
            subdataset_indices.extend(global_indices[class_remaining_indices])
        subdataset = AugmentedSubset(Subset(training_dataset, subdataset_indices))

        pruned_indices = list(set(range(len(training_dataset))) - set(subdataset_indices))
        pruned_dataset = AugmentedSubset(Subset(training_dataset, pruned_indices))

        self.fig_save_dir = os.path.join(self.fig_save_dir, 'clp' + str(int(round(self.prune_percentage * 100))),
                                         self.dataset_name)
        self.res_save_dir = os.path.join(self.res_save_dir, 'clp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.plot_class_level_sample_distribution(subdataset_indices, 'clp', labels)
        self.fig_save_dir = os.path.join(ROOT, 'Figures/')
        self.res_save_dir = os.path.join(ROOT, 'Results/')

        return subdataset, pruned_dataset

    def compute_el2n_scores(self, model: ResNet18LowRes, dataloader: DataLoader) -> Tuple[List[float], List[int]]:
        """
        Compute EL2N scores for all samples in the dataset given the model.
        EL2N = L2 norm between predicted probability vector and one-hot label vector.

        :param model: Trained PyTorch model
        :param dataloader: DataLoader for the dataset

        :return: List of EL2N scores and the labels of each data sample
        """
        sample_el2n, all_labels = [], []

        model.eval()
        with torch.no_grad():
            for inputs, labels, _ in dataloader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                probs = functional.softmax(outputs, dim=1)

                for i in range(len(inputs)):
                    label = labels[i].item()
                    all_labels.append(label)

                    prob = probs[i]
                    one_hot = functional.one_hot(torch.tensor(label), num_classes=self.num_classes).float().to(DEVICE)
                    el2n = torch.norm(prob - one_hot).item()
                    sample_el2n.append(el2n)
        return sample_el2n, all_labels

    def compute_class_level_scores(self, el2n_scores: List[List[float]], labels: List[int]) -> Dict[int, List[float]]:
        """
        Aggregate EL2N scores across models and compute per-class hardness estimates.

        :param el2n_scores: List of lists [num_models][num_samples]
        :param labels: List of labels [num_samples]

        :return: Dict mapping class_id -> average EL2N score for that class
        """
        # Average across models → [num_samples]
        el2n_mean = np.mean(np.array(el2n_scores), axis=0)

        # Compute class-level scores
        class_level_scores = {class_id: [] for class_id in range(self.num_classes)}
        for score, label in zip(el2n_mean, labels):
            class_level_scores[label].append(score)

        return class_level_scores

    def resampling_pruned_subdataset(self, oversampling_strategy: str, labels: List[int],
                                     training_dataset: Union[AugmentedSubset, IndexedDataset],
                                     hardness_sorted_by_class: Union[None, Dict[int, List[float]]]) -> AugmentedSubset:
        """
        Removes the specified percentage of samples from the dataset. Produces imbalanced subdatasets. Works by firstly
        performing class_Level_pruning, and later performing hardness_based resampling on the pruned subdataset to
        introduce the data imbalance. Uses easy pruning as a default undersampling strategy.

        :param oversampling_strategy: Name of the oversampling strategy for hardness-based resampling.
        :param labels: List of labels for each data sample in the original dataset.
        :param training_dataset: Original training dataset
        :param hardness_sorted_by_class: Instance-level hardness estimates used to guide pruning. If these are set to
        None than EL2N scores are computed from the pruned subdatasets.

        :return: Imbalanced pruned subdataset
        """
        subdataset, pruned_dataset = self.class_level_pruning(labels, training_dataset)

        if hardness_sorted_by_class is None:
            sub_dataloader = get_dataloader(subdataset, 1000)
            trainer = ModelTrainer(len(subdataset), [sub_dataloader], None, self.dataset_name, stop_at_probe=True)
            el2n_scores, sub_labels = [], None
            for model_id in tqdm.tqdm(range(self.num_models_per_dataset)):
                model = trainer.train_model(0, model_id, None)
                scores, sub_labels = self.compute_el2n_scores(model, sub_dataloader)
                el2n_scores.append(scores)
            hardness_sorted_by_class = self.compute_class_level_scores(el2n_scores, sub_labels)

        resampler = DataResampling(subdataset, self.num_classes, oversampling_strategy, 'easy',
                                   hardness_sorted_by_class, True, self.dataset_name, self.num_models_for_hardness,
                                   self.mean, self.std, pruned_dataset)
        resampled_dataset = resampler.resample_data(self.imbalance_ratio)

        print('II. prune_percentage', self.prune_percentage)
        self.fig_save_dir = os.path.join(self.fig_save_dir,
                                         f'{oversampling_strategy}_dlp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        self.res_save_dir = os.path.join(self.res_save_dir,
                                         f'{oversampling_strategy}_dlp' + str(int(self.prune_percentage * 100)),
                                         self.dataset_name)
        subdataset_indices = [idx for _, _, idx in resampled_dataset]
        self.plot_class_level_sample_distribution(subdataset_indices, 'dlp', labels)

        return resampled_dataset

# TODO: Make it so that the imbalance_ratio can be computed from EL2N.
