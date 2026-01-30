"""Core module for pruning the dataset that returns the pruned subdataset."""

from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Subset
import tqdm

from config import DEVICE, get_config
from data import AugmentedSubset, get_dataloader, IndexedDataset
from data_resampling import DataResampling
from neural_networks import ResNet18LowRes
from train_ensemble import ModelTrainer


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

    def class_level_pruning(
            self,
            labels: List[int],
            training_dataset: Union[AugmentedSubset, IndexedDataset],
            hardness_sorted_by_class: Dict[int, List[Tuple[int, float]]]
    ) -> Tuple[AugmentedSubset, AugmentedSubset, Dict[int, List[float]]]:
        """
        Remove the specified percentage of samples from each class. Ensures that the class distribution remains
        balanced.

        :param labels: List of labels for each data sample in the original dataset.
        :param training_dataset: Original training dataset
        :param hardness_sorted_by_class: Instance-level hardness estimates (and the corresponding sample indices) sorted
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

        pruned_indices = list(set(range(len(training_dataset))) - set(subdataset_indices))
        pruned_dataset = AugmentedSubset(Subset(training_dataset, pruned_indices))
        hardness_sorted_by_class = {c: [hardness_sorted_by_class[c][i][1]
                                        for i in range(len(hardness_sorted_by_class[c]))
                                        if hardness_sorted_by_class[c][i][0] in subdataset_indices]
                                    for c in hardness_sorted_by_class.keys()}

        return subdataset, pruned_dataset, hardness_sorted_by_class

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

    def prune_and_resample(
            self,
            oversampling_strategy: str,
            labels: List[int],
            training_dataset: Union[AugmentedSubset, IndexedDataset],
            hardness_sorted_by_class: Dict[int, List[Tuple[int, float]]]
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
        :param hardness_sorted_by_class: Instance-level hardness estimates together with corresponding sample indices
        used to guide pruning. If these are set to None then EL2N scores are computed from the pruned subdatasets.

        :return: Imbalanced pruned subdataset
        """
        subdataset, pruned_dataset, hardness_sorted_by_class = self.class_level_pruning(labels, training_dataset,
                                                                                        hardness_sorted_by_class)

        # The below allows to use EL2N (computed on the pruned subdatasets) instead of AUM to guide pruning.
        if hardness_sorted_by_class is None:
            sub_dataloader = get_dataloader(subdataset, 1000)
            trainer = ModelTrainer(len(subdataset), [sub_dataloader], None, self.dataset_name, stop_at_probe=True)
            el2n_scores, sub_labels = [], None
            for model_id in tqdm.tqdm(range(self.num_models_per_dataset)):
                model = trainer.train_model(0, model_id, None)
                scores, sub_labels = self.compute_el2n_scores(model, sub_dataloader)
                el2n_scores.append(scores)
            hardness_sorted_by_class = self.compute_class_level_scores(el2n_scores, sub_labels)

        if oversampling_strategy != "none":
            resampler = DataResampling(subdataset, self.num_classes, oversampling_strategy, 'easy',
                                       hardness_sorted_by_class, True, self.dataset_name, self.num_models_for_hardness,
                                       self.mean, self.std, pruned_dataset)
            subdataset = resampler.resample_data(self.imbalance_ratio)

        return subdataset
