import os
import pickle
import random
import re
from typing import List

import numpy as np
import torch

from config import ROOT, get_config


def set_reproducibility(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_latest_model_index(save_dir, num_epochs):
    """Find the latest model index from saved files in the save directory. This makes it easier to add more models
    to the ensemble, as we don't have to retrain from scratch."""
    max_index = -1
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            match = re.search(rf'model_(\d+)_epoch_{num_epochs}\.pth$', filename)
            if match:
                index = int(match.group(1))
                max_index = max(max_index, index)
    return max_index


def load_results(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def load_aum_results(data_cleanliness, dataset_name, num_epochs) -> List[List[float]]:
    """Loading the AUM results and changing their format to match that of other hardness estimators by summing over
    epochs."""
    aum_path = os.path.join(ROOT, f'Results/{data_cleanliness}{dataset_name}', 'AUM.pkl')
    aum_over_epochs_and_models = load_results(aum_path)

    aum_scores = [
        [
            sum(model_list[sample_idx][epoch_idx] for epoch_idx in range(num_epochs)) / num_epochs
            for sample_idx in range(len(aum_over_epochs_and_models[0]))
        ]
        for model_list in aum_over_epochs_and_models
    ]

    return aum_scores


def load_forgetting_results(data_cleanliness, dataset_name) -> List[List[int]]:
    forgetting_path = os.path.join(ROOT, f'Results/{data_cleanliness}{dataset_name}', 'Forgetting.pkl')
    forgetting_scores = load_results(forgetting_path)

    return forgetting_scores
