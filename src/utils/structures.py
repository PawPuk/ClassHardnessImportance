from collections import defaultdict
import os
import re
from typing import List


def get_latest_model_index(save_dir: str, num_epochs: int, max_dataset_count: int) -> List[int]:
    """Find the latest trained model index for each version of the dataset. This makes it easier to add more models
    to the ensembles, as we don't have to retrain from scratch."""
    max_indices = defaultdict(lambda: -1)  # -1 means that the next index is 0
    if os.path.exists(save_dir):
        for filename in os.listdir(save_dir):
            match = re.search(rf'dataset_(\d+)_model_(\d+)_epoch_{num_epochs}\.pth$', filename)
            if match:
                dataset_idx = int(match.group(1))
                model_idx = int(match.group(2))
                max_indices[dataset_idx] = max(max_indices[dataset_idx], model_idx)
    return [max_indices[i] for i in range(max_dataset_count)]


def defaultdict_to_dict(d):
    """Recursively convert defaultdicts to dicts at all depths."""
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):  # catch plain dicts too
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d
