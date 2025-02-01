import os
import pickle
import random
import re
from typing import List

import numpy as np
import torch


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


def load_aum_results(hardness_save_dir, num_epochs) -> List[List[float]]:
    """Loading the AUM results and changing their format to match that of other hardness estimators by summing over
    epochs."""
    aum_path = os.path.join(hardness_save_dir, 'AUM.pkl')
    aum_over_epochs_and_models = load_results(aum_path)

    # TODO: This is CURRENTLY required as train_ensemble.py wasn't initially working properly with denoised datasets.
    for model_idx, model_list in enumerate(aum_over_epochs_and_models):
        aum_over_epochs_and_models[model_idx] = [sample for sample in model_list if len(sample) > 0]

    aum_scores = [
        [
            sum(model_list[sample_idx][epoch_idx] for epoch_idx in range(num_epochs)) / num_epochs
            for sample_idx in range(len(aum_over_epochs_and_models[0]))
        ]
        for model_list in aum_over_epochs_and_models
    ]

    return aum_scores


def load_forgetting_results(hardness_save_dir, num_samples) -> List[List[float]]:
    forgetting_path = os.path.join(hardness_save_dir, 'Forgetting.pkl')
    forgetting_scores = load_results(forgetting_path)

    # TODO: This is CURRENTLY required as train_ensemble.py wasn't initially working properly with denoised datasets.
    forgetting_scores = [model_list[:num_samples] for model_list in forgetting_scores]
    return forgetting_scores
