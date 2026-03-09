from collections import defaultdict
import os
from typing import Any, Dict, List, Tuple

import torch

from src.config.config import ROOT
from src.utils.structures import defaultdict_to_dict


def load_model_states(dataset_name: str, num_models_for_hardness: int, num_epochs: int) -> List[Any]:
    """Load the pre-trained models."""
    models_dir = os.path.join(ROOT, "Models")
    model_states = []
    full_dataset_dir = os.path.join(models_dir, "none", dataset_name)

    for file in os.listdir(full_dataset_dir):
        if len(model_states) < num_models_for_hardness and file.endswith(".pth") and f"_epoch_{num_epochs}" in file:
            model_path = os.path.join(full_dataset_dir, file)
            model_state = torch.load(model_path)
            model_states.append(model_state)

    print(f"Loaded {len(model_states)} models for estimating confidence.")
    return model_states


def load_models_from_cs2(dataset_name: str, num_epochs: int, num_datasets: int,
                         num_models_per_dataset: int) -> Dict[Tuple[str, str], Dict[int, Dict[int, List[str]]]]:
    """Used to load models trained in the second case study (resampling on full datasets)."""
    models_dir = os.path.join(ROOT, "Models")
    models_by_strategy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for root, dirs, files in os.walk(models_dir):
        if 'over_' in root and os.path.basename(root) == dataset_name:
            oversampling_strategy = root.split("over_")[1].split("_under_")[0]
            undersampling_strategy = root.split("_under_")[1].split("_alpha_")[0]
            alpha = root.split("_alpha_")[1].split("_hardness_")[0]
            key = (oversampling_strategy, undersampling_strategy)
            if alpha == 3 and dataset_name == 'CIFAR100':
                continue

            for file in files:
                if file.endswith(".pth") and f"_epoch_{num_epochs}" in file:
                    # For cases where the number of trained models is above robust_ensemble_size
                    dataset_index = int(file.split("_")[1])
                    model_index = int(file.split("_")[3])
                    if dataset_index >= num_datasets or model_index >= num_models_per_dataset:
                        raise Exception('The `num_datasets` and `num_models_per_dataset` in config.py needs to '
                                        'have the same values as it when running experiment3.py.')

                    model_path = os.path.join(root, file)
                    models_by_strategy[key][alpha][dataset_index].append(model_path)
            if len(models_by_strategy[key][alpha]) > 0:
                print(f"Loaded {len(models_by_strategy[key][alpha])} ensembles of models for strategies {key} and "
                      f"alpha {alpha}, with each ensemble having {len(models_by_strategy[key][alpha][0])} models.")
                for i in range(1, len(models_by_strategy[key][alpha])):  # Sanity check
                    assert len(models_by_strategy[key][alpha][0]) == len(models_by_strategy[key][alpha][i])

    # Also load models trained on the full dataset (no resampling)
    full_dataset_dir = os.path.join(models_dir, "none", dataset_name)
    if os.path.exists(full_dataset_dir):
        key = ('none', 'none')
        for file in os.listdir(full_dataset_dir):
            if file.endswith(".pth") and f"_epoch_{num_epochs}" in file:
                model_path = os.path.join(full_dataset_dir, file)
                models_by_strategy[key][1][0].append(model_path)

    print(f"Loaded {len(models_by_strategy.keys())} ensembles for {dataset_name}.")
    return defaultdict_to_dict(models_by_strategy)


def load_models_from_cs1(dataset_name: str, num_epochs: int, num_datasets: int, num_models_per_dataset: int,
                         alpha: int = 0) -> Dict[str, Dict[int, Dict[int, List[str]]]]:
    """Used to load models trained in the first case study (resampling on pruned datasets)."""
    models_dir = os.path.join(ROOT, "Models/")
    models_by_strategy, pruning_rate = defaultdict(lambda: defaultdict(lambda: defaultdict(list))), None

    for pruning_strategy in ['none', 'random', 'holdout', 'SMOTE']:
        # Walk through each folder in the Models directory
        for root, dirs, files in os.walk(models_dir):
            # Ensure the dataset name matches exactly (avoid partial matches like "cifar10" in "cifar100")
            if f"{pruning_strategy}_pruning" in root and os.path.basename(root) == dataset_name:
                pruning_rate = int(root.split("pruning_rate_")[1].split("_alpha_")[0])
                model_alpha = int(root.split("_alpha_")[1].split("/")[0])

                for file in files:
                    if file.endswith(".pth") and f"_epoch_{num_epochs}" in file and model_alpha == alpha:
                        dataset_index = int(file.split("_")[1])
                        model_index = int(file.split("_")[3])
                        if dataset_index >= num_datasets or model_index >= num_models_per_dataset:
                            raise Exception('The `num_datasets` and `num_models_per_dataset` in config.py needs to '
                                            'have the same values as it when running experiment3.py.')

                        model_path = os.path.join(root, file)
                        models_by_strategy[pruning_strategy][pruning_rate][dataset_index].append(model_path)
                if len(models_by_strategy[pruning_strategy][pruning_rate]) > 0:
                    for i in range(1, len(models_by_strategy[pruning_strategy][pruning_rate])):  # Sanity check
                        assert len(models_by_strategy[pruning_strategy][pruning_rate][0]) == \
                               len(models_by_strategy[pruning_strategy][pruning_rate][i])
                    print(
                        f"Loaded {len(models_by_strategy[pruning_strategy][pruning_rate])} ensembles of models for "
                        f"strategies {pruning_strategy} and pruning rate {pruning_rate}, with each ensemble having "
                        f"{len(models_by_strategy[pruning_strategy][pruning_rate][0])} models.")

        print(f"Models loaded by pruning rate for {pruning_strategy} on {dataset_name}")

    print(models_by_strategy.keys())
    print(models_by_strategy)
    return defaultdict_to_dict(models_by_strategy)
