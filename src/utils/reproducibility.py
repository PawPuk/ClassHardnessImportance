"""Contains all the functions relating to setting or computing seeds."""

import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch


def compute_current_seed(
        config: Dict[str, Union[int, float, str, List[int], List[float], List[str], Tuple[float, float, float]]],
        current_dataset_index: int,
        current_model_index: int
) -> int:
    """Compute the seed for training the current model."""
    base_seed = config['probe_base_seed']
    seed_step = config['probe_seed_step']
    dataset_step = config['probe_dataset_step']

    seed = base_seed + current_dataset_index * dataset_step + current_model_index * seed_step
    return seed


def set_reproducibility(seed: int = 42):
    """Sets the seed to specific value for all random events."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = False
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.deterministic = True
