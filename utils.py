from collections import defaultdict
import os
import pickle
import random
import re

import numpy as np
import torch

from config import ROOT


def set_reproducibility(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_latest_model_index(save_dir, num_epochs, max_dataset_count):
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


def load_results(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def load_hardness_estimates(data_cleanliness, dataset_name):
    path = os.path.join(ROOT, f'Results/{data_cleanliness}{dataset_name}', 'hardness_estimates.pkl')
    hardness_estimates = load_results(path)
    return hardness_estimates