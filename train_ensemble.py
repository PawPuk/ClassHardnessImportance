"""Core module that allows for training ensembles of models as well as estimating hardness."""

import os
import pickle
from typing import cast, Dict, List, Sized, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import DEVICE, get_config, ROOT
from neural_networks import ResNet18LowRes
from utils import set_reproducibility, get_latest_model_index


class ModelTrainer:
    """Allows training ensembles of models as well as estimating hardness."""
    def __init__(self, training_set_size: int, training_loaders: List[DataLoader], test_loader: Union[DataLoader, None],
                 dataset_name: str, pruning_type: str = 'none', save_probe_models: bool = True,
                 stop_at_probe: bool = False, estimate_hardness: bool = False, clean_data: bool = False):
        """
        Initialize the ModelTrainer class with configuration specific to the dataset.

        :param training_set_size: Specified the size of the training set
        :param training_loaders: List of DataLoaders for the training datasets. For experiment1.py where only one
        dataset is used pass the DataLoader in a List.
        :param test_loader: DataLoader for the test dataset.
        :param dataset_name: The name of the dataset being used.
        :param pruning_type: Type of pruning being applied (default: 'none'). This is used in experiment2.py and
        experiment3.py to ensure unique saving directories.
        :param save_probe_models: Whether to save the probe models after a specified epoch (default: True). This is
        required for EL2N computation.
        :param stop_at_probe: Set to yes to stop training after obtaining the probe network (useful when we only want
        to estimate hardness through EL2N and not fully train).
        :param estimate_hardness: Specify if the hardness should be saved and stored during training (default False). We
        set this to True only for experiment1.py as we do not currently estimate hardness on pruned or resampled
        datasets.
        :param clean_data: Allows differentiating between models trained on clean and noisy data while saving. This is
        used to ensure unique saving directories.
        """
        self.training_set_size = training_set_size
        self.training_loaders = training_loaders
        self.test_loader = test_loader
        self.pruning_type = pruning_type
        self.dataset_name = dataset_name
        self.save_probe_models = save_probe_models
        self.stop_at_probe = stop_at_probe
        self.estimate_hardness = estimate_hardness
        self.clean_data = 'clean' if clean_data else 'unclean'

        self.config = get_config(self.dataset_name)

        # Incorporate dataset_name and pruning_type into directories to prevent overwriting
        self.num_epochs = self.config['num_epochs']
        self.dataset_count = self.config['num_datasets']
        self.save_dir = os.path.join(self.config['save_dir'], pruning_type, f"{self.clean_data}{dataset_name}")
        os.makedirs(self.save_dir, exist_ok=True)

    def compute_current_seed(self, current_dataset_index: int, current_model_index: int) -> int:
        """Compute the seed for training the current model."""
        base_seed = self.config['probe_base_seed']
        seed_step = self.config['probe_seed_step']
        dataset_step = self.config['probe_dataset_step']

        seed = base_seed + current_dataset_index * dataset_step + current_model_index * seed_step
        return seed

    @staticmethod
    def estimate_instance_hardness(
            batch_indices: torch.Tensor,
            inputs: torch.Tensor,
            outputs: torch.Tensor,
            labels: torch.Tensor,
            predicted: torch.Tensor,
            hardness_estimates: Dict[Tuple[int, int], Dict[str, List[Union[float, List[Union[None, float]]]]]],
            epoch: int,
            remembering: List[bool],
            dataset_model_id: Tuple[int, int]
    ):
        """Used to produce model-based hardness estimates of every data sample in the given batch."""
        for index_within_batch, (i, x, logits, correct_label) in enumerate(zip(batch_indices, inputs, outputs, labels)):
            i = i.item()
            correct_label = correct_label.item()
            predicted_label = predicted[index_within_batch].item()

            logits = logits.detach()
            correct_logit = logits[correct_label].item()
            probs = torch.nn.functional.softmax(logits, dim=0)
            # Confidence
            hardness_estimates[dataset_model_id]['Confidence'][i][epoch] = correct_logit
            # AUM
            max_other_logit = torch.max(torch.cat((logits[:correct_label], logits[correct_label + 1:]))).item()
            hardness_estimates[dataset_model_id]['AUM'][i][epoch] = correct_logit - max_other_logit
            # DataIQ
            p_y = probs[correct_label].item()
            hardness_estimates[dataset_model_id]['DataIQ'][i][epoch] = p_y * (1 - p_y)
            # Cross-Entropy Loss
            label_tensor = torch.tensor([correct_label], device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), label_tensor).item()
            hardness_estimates[dataset_model_id]['Loss'][i][epoch] = loss
            # Forgetting
            if predicted_label == correct_label:
                remembering[i] = True
            elif predicted_label != correct_label and remembering[i]:
                hardness_estimates[dataset_model_id]['Forgetting'][i] += 1
                remembering[i] = False

    def evaluate_model(self, model: ResNet18LowRes, criterion: nn.CrossEntropyLoss) -> Tuple[float, float]:
        """Evaluate the model on the test set."""
        model.eval()
        correct, total, running_loss = 0, 0, 0.0

        with torch.no_grad():
            for inputs, labels, _ in self.test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = running_loss / total
        return avg_loss, accuracy

    def train_model(self, current_dataset_index: int, current_model_index: int,
                    hardness_estimates: Union[Dict[Tuple[int, int], Dict], None]) -> Union[None, ResNet18LowRes]:
        """Train a single model."""
        dataset_model_id = (current_dataset_index, current_model_index)
        seed = self.compute_current_seed(current_dataset_index, current_model_index)
        set_reproducibility(seed)

        model = ResNet18LowRes(num_classes=self.config['num_classes']).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'],
                              weight_decay=self.config['weight_decay'], nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config['lr_decay_milestones'], gamma=0.2)

        if self.estimate_hardness:
            for estimator in ['Confidence', 'AUM', 'DataIQ', 'Loss']:   # [epoch_index][sample_index]
                hardness_estimates[dataset_model_id][estimator] = [[None for _ in range(self.num_epochs)]
                                                                   for _ in range(self.training_set_size)]
            hardness_estimates[dataset_model_id]['Forgetting'] = [0 for _ in range(self.training_set_size)]
        remembering = [False for _ in range(self.training_set_size)]

        for epoch in range(self.config['num_epochs']):
            model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            for inputs, labels, indices in self.training_loaders[current_dataset_index]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

                if self.estimate_hardness:
                    self.estimate_instance_hardness(indices, inputs, outputs, labels, predicted, hardness_estimates,
                                                    epoch, remembering, dataset_model_id)
            scheduler.step()

            if self.test_loader is not None:
                avg_train_loss = running_loss / total_train
                train_accuracy = 100 * correct_train / total_train
                avg_test_loss, test_accuracy = self.evaluate_model(model, criterion)
                print(f'Model {current_model_index}, '
                      f'Epoch [{epoch + 1}/{self.config["num_epochs"]}] '
                      f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                      f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

            # Optionally save the model at a specific epoch
            if epoch + 1 == self.config['save_epoch']:
                if self.stop_at_probe:
                    return model
                if self.save_probe_models:
                    save_path = os.path.join(self.save_dir, f'dataset_{current_dataset_index}_model_'
                                                            f'{current_model_index}'
                                                            f'_epoch_{epoch + 1}.pth')
                    torch.save(model.state_dict(), save_path)
                    print(f'Model {current_model_index} ({self.dataset_name}, '
                          f'{self.pruning_type} dataset) saved at epoch {self.config["save_epoch"]}.')

        # Save model after full training
        final_save_path = os.path.join(self.save_dir, f'dataset_{current_dataset_index}'
                                                      f'_model_{current_model_index}'
                                                      f'_epoch_{self.config["num_epochs"]}.pth')
        torch.save(model.state_dict(), final_save_path)
        print(f'Dataset {current_dataset_index} Model {current_model_index} '
              f'({self.dataset_name}, {self.pruning_type} dataset) saved after full training at epoch '
              f'{self.config["num_epochs"]}.')
        return None

    @staticmethod
    def load_previous_hardness_estimates(path: str) -> Union[Dict, Dict[Tuple[int, int], Dict[str, List[float]]]]:
        """Loads the hardness estimates, if they have been computed before, or return an empty Dictionary otherwise."""
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as file:
                prior_hardness_estimates = pickle.load(file)
            print(f'{path} exists - extended hardness estimates.')
            return prior_hardness_estimates
        else:
            print(f"{path} does not exist or is empty. Initializing new data.")
            return {}

    def save_results(self, hardness_estimates: Dict[Tuple[int, int], Dict[str, List[float]]],
                     dataset_model_id: Tuple[int, int]):
        """
        The purpose of this function is to enable easier generation of results. If we already spent a lot of
        resources on training an ensemble, we don't want it to go to waste just because the ensemble is not large
        enough. We want to add more models to the ensemble rather than have to retrain it from scratch.
        """
        hardness_save_dir = os.path.join(ROOT, f"Results/{self.clean_data}{self.dataset_name}/")
        os.makedirs(hardness_save_dir, exist_ok=True)
        path = os.path.join(hardness_save_dir, 'hardness_estimates.pkl')
        old_hardness_estimates = self.load_previous_hardness_estimates(path)
        old_hardness_estimates[dataset_model_id] = hardness_estimates[dataset_model_id]

        with open(path, "wb") as file:
            print(f'Saving updated hardness estimates.')
            # noinspection PyTypeChecker
            pickle.dump(old_hardness_estimates, file)

    def train_ensemble(self):
        """Train an ensemble of models."""

        latest_model_indices = get_latest_model_index(self.save_dir, self.config['num_epochs'], self.dataset_count)
        num_models_to_train_per_dataset = self.config['num_models_per_dataset']

        print(f"Starting training {self.dataset_count} ensembles of {num_models_to_train_per_dataset} models on "
              f"{self.dataset_name}.")
        print(f"Number of samples in the training loader: {len(cast(Sized, self.training_loaders[0].dataset))}")
        print(f"Number of samples in the test loader: {len(cast(Sized, self.test_loader.dataset))}")
        print('-'*20)

        for dataset_id in tqdm(range(self.dataset_count)):
            for model_id in tqdm(range(latest_model_indices[dataset_id] + 1, num_models_to_train_per_dataset)):
                hardness_estimates = {(dataset_id, model_id): {}}
                self.train_model(dataset_id, model_id, hardness_estimates)
                if self.estimate_hardness:
                    for estimator in ['Confidence', 'AUM', 'DataIQ', 'Loss']:
                        hardness_estimates[(dataset_id, model_id)][estimator] = np.mean(
                            hardness_estimates[(dataset_id, model_id)][estimator], axis=1)
                    self.save_results(hardness_estimates, (dataset_id, model_id))
