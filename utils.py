"""Module that contains a few useful functions."""

from collections import defaultdict
import os
import pickle
import random
import re
from typing import Dict, List, Tuple

import numpy as np
import torch

from config import ROOT


def set_reproducibility(seed: int=42):
    """Crucial for ensuring code reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = False
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.deterministic = True


def get_latest_model_index(save_dir: str, num_epochs: int, max_dataset_count: int) -> List[int]:
    """Find the latest model index from saved files in the save directory. This makes it easier to add more models
    to the ensemble, as we don't have to retrain from scratch."""
    max_indices = defaultdict(lambda: -1)
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            match = re.search(rf'dataset_(\d+)model_(\d+)_epoch_{num_epochs}\.pth$', filename)
            if match:
                dataset_idx = int(match.group(1))
                model_idx = int(match.group(2))
                max_indices[dataset_idx] = max(max_indices[dataset_idx], model_idx)
    return [max_indices[i] for i in range(max_dataset_count)]


def load_results(path: str):
    """Load results."""
    with open(path, 'rb') as file:
        return pickle.load(file)


def load_hardness_estimates(data_cleanliness: str, dataset_name: str) -> Dict[Tuple[int, int], Dict[str, List[float]]]:
    """Load hardness estimates."""
    path = os.path.join(ROOT, f'Results/{data_cleanliness}{dataset_name}', 'hardness_estimates.pkl')
    hardness_estimates = load_results(path)
    return hardness_estimates

def compute_sample_allocation_after_resampling(hardness_scores: List[float], labels: List[int], num_classes: int,
                                               num_training_samples: int, hardness_estimator: str,
                                               pruning_rate: int = 0, alpha: int = 1
                                               ) -> Tuple[List[int], Dict[int, List[float]]]:
    """Compute number of samples per class after hardness-based resampling according to hardness_scores."""
    hardnesses_by_class = {class_id: [] for class_id in range(num_classes)}

    # Divide the hardness estimates into classes.
    print(len(labels), len(hardness_scores))
    for i, label in enumerate(labels):
        hardnesses_by_class[label].append(hardness_scores[i])

    # Compute average hardness of each class.
    means_hardness_by_class = {class_id: np.mean(vals) for class_id, vals in hardnesses_by_class.items()}

    # Add offset in case some of the classes have negative hardness values to not get nonsensical resampling ratios.
    if min(means_hardness_by_class.values()) < 0:
        offset = -min(means_hardness_by_class.values())
        for class_id in range(num_classes):
            means_hardness_by_class[class_id] += offset

    # Compute the resampling ratios for each class.
    if hardness_estimator in ['AUM', 'Confidence', 'iAUM', 'iConfidence']:
        hardness_ratios = {class_id: 1 / float(val) for class_id, val in means_hardness_by_class.items()}
    else:
        hardness_ratios = {class_id: float(val) for class_id, val in means_hardness_by_class.items()}
    ratios = {class_id: class_hardness / sum(hardness_ratios.values())
              for class_id, class_hardness in hardness_ratios.items()}

    # Compute the amount of samples per class after resampling.
    samples_per_class = [int(round((1 - pruning_rate / 100) * ratio * num_training_samples))
                         for class_id, ratio in ratios.items()]

    # Tailor the degree of the introduces data imbalance (only applicable if alpha is larger than 1).
    if alpha > 1:
        average_sample_count = int(np.mean(samples_per_class))
        for class_id in range(num_classes):
            absolute_difference = abs(samples_per_class[class_id] - average_sample_count)
            if samples_per_class[class_id] > average_sample_count:
                samples_per_class[class_id] = average_sample_count + int(alpha * absolute_difference)
            else:
                samples_per_class[class_id] = average_sample_count - int(alpha * absolute_difference)

    return samples_per_class, hardnesses_by_class