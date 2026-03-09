import os
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np

from src.config.config import ROOT


def load_results(path: str):
    """Load results."""
    with open(path, 'rb') as file:
        return pickle.load(file)


def load_hardness_estimates(dataset_name: str, hardness_estimator: str, num_models_for_hardness: int) -> List[float]:
    """Load hardness estimates (output only for specific hardness estimator and a single scalar per data sample)."""
    path = os.path.join(ROOT, 'Results', dataset_name, 'hardness_estimates.pkl')
    hardness_estimates = load_results(path)
    hardness_over_models = [hardness_estimates[(0, model_id)][hardness_estimator]
                            for model_id in range(len(hardness_estimates))]
    n = min(len(hardness_estimates), num_models_for_hardness)
    final_hardness_estimates = list(np.mean(np.array(hardness_over_models[:n]), axis=0))
    return final_hardness_estimates


def load_previous_hardness_estimates(path: str) -> Union[Dict, Dict[Tuple[int, int], Dict[str, List[float]]]]:
    """Loads the hardness estimates, if they have been computed before, or return an empty Dictionary otherwise."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        prior_hardness_estimates = load_results(path)
        print(f'{path} exists - extended hardness estimates.')
        return prior_hardness_estimates
    else:
        print(f"{path} does not exist or is empty. Initializing new data.")
        return {}


def load_sample_allocations(hardness_save_dir: str, dataset_name: str) -> Dict[int, List[int]]:
    """Extracts the sample allocations after resampling for different alphas"""
    sample_allocations = {}
    for root, dirs, files in os.walk(hardness_save_dir):
        if f"{dataset_name}/" in root and 'alpha_' in root:
            alpha = int(root.split('alpha_')[-1])
            for file in files:
                file_path = os.path.join(root, file)
                sample_allocations[alpha] = load_results(file_path)
    print(f'Loaded info on class-wise sample allocation after hardness-based resampling:\n\t{sample_allocations}')
    return sample_allocations


def save_results(hardness_estimates: Dict[Tuple[int, int], Dict[str, List[float]]], dataset_model_id: Tuple[int, int],
                 dataset_name: str):
    """
    The purpose of this function is to enable easier generation of results. If we already spent a lot of
    resources on training an ensemble, we don't want it to go to waste just because the ensemble is not large
    enough. We want to add more models to the ensemble rather than have to retrain it from scratch.
    """
    hardness_save_dir = os.path.join(ROOT, "Results", dataset_name)
    os.makedirs(hardness_save_dir, exist_ok=True)
    path = os.path.join(hardness_save_dir, 'hardness_estimates.pkl')
    old_hardness_estimates = load_previous_hardness_estimates(path)
    old_hardness_estimates[dataset_model_id] = hardness_estimates[dataset_model_id]

    with open(path, "wb") as file:
        print(f'Saving updated hardness estimates.')
        # noinspection PyTypeChecker
        pickle.dump(old_hardness_estimates, file)
