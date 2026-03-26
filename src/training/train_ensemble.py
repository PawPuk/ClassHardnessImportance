"""Core module that allows for training ensembles of models as well as estimating hardness."""

import os
from typing import cast, Dict, List, Sized, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config.config import DEVICE, get_config
from src.hardness.estimators import estimate_instance_hardness
from src.models.neural_networks import ResNet18LowRes
from src.utils.evaluation import evaluate_model
from src.utils.io import save_results
from src.utils.reproducibility import compute_current_seed, set_reproducibility
from src.utils.structures import get_latest_model_index


class ModelTrainer:
    """Allows training ensembles of models as well as estimating hardness."""
    def __init__(
            self,
            training_set_size: int,
            training_loaders: List[DataLoader],
            test_loader: Union[DataLoader, None],
            dataset_name: str,
            pruning_type: str = 'none',
            save_probe_models: bool = True,
            estimate_hardness: bool = False,
            for_experiment_1: bool = False
    ):
        """
        Initialize the ModelTrainer class with configuration specific to the dataset.

        :param training_set_size: Specified the size of the training set. This is only useful for experiment1.py.
        :param training_loaders: List of DataLoaders for the training datasets. For experiment1.py where only one
        dataset is used pass the DataLoader in a List.
        :param test_loader: DataLoader for the test dataset.
        :param dataset_name: The name of the dataset being used.
        :param pruning_type: Type of pruning being applied (default: 'none'). This is used in experiment2.py and
        experiment3.py to ensure unique saving directories.
        :param save_probe_models: Whether to save the probe models after a specified epoch (default: True). This is
        required for EL2N computation.
        :param estimate_hardness: Specify if the hardness should be saved and stored during training (default False). We
        set this to True only for experiment1.py as we do not currently estimate hardness on pruned or resampled
        datasets.
        """
        self.training_set_size = training_set_size
        self.training_loaders = training_loaders
        self.test_loader = test_loader
        self.pruning_type = pruning_type
        self.dataset_name = dataset_name
        self.save_probe_models = save_probe_models
        self.estimate_hardness = estimate_hardness

        self.config = get_config(self.dataset_name)

        self.num_epochs = self.config['num_epochs']
        # For experiment1.py we train single ensemble as there is only one dataset (unlike experiment2.py or
        # experiment3.py where we train on multiple versions of a dataset to account for variability in its creation)
        if for_experiment_1:
            self.num_models_to_train_per_dataset = self.config['num_datasets'] * self.config['num_models_per_dataset']
            self.dataset_count = 1
        else:
            self.num_models_to_train_per_dataset = self.config['num_models_per_dataset']
            self.dataset_count = self.config['num_datasets']

        self.save_dir = os.path.join(self.config['save_dir'], pruning_type, dataset_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def train_model(
            self,
            current_dataset_index: int,
            current_model_index: int,
            hardness_estimates: Union[Dict[Tuple[int, int], Dict], None]
    ) -> Union[None, ResNet18LowRes]:
        """Train a single model."""
        dataset_model_id = (current_dataset_index, current_model_index)
        seed = compute_current_seed(self.config, current_dataset_index, current_model_index)
        set_reproducibility(seed)

        model = ResNet18LowRes(num_classes=self.config['num_classes']).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'],
                              weight_decay=self.config['weight_decay'], nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config['lr_decay_milestones'], gamma=0.2)

        if self.estimate_hardness:
            for estimator in ['Confidence', 'AUM', 'DataIQ', 'Loss']:
                # hardness_estimates[dataset_model_id][estimator][epoch_index][sample_index]: float
                hardness_estimates[dataset_model_id][estimator] = [[0.0 for _ in range(self.num_epochs)]
                                                                   for _ in range(self.training_set_size)]
            # hardness_estimates[dataset_model_id]['Forgetting'][sample_index]: int
            hardness_estimates[dataset_model_id]['Forgetting'] = [0 for _ in range(self.training_set_size)]
        remembering = [False for _ in range(self.training_set_size)]  # Required to computing Forgetting

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
                    estimate_instance_hardness(indices, inputs, outputs, labels, predicted, hardness_estimates, epoch,
                                               remembering, dataset_model_id)
            scheduler.step()

            # Report progress (accuracy & loss on training & test sets)
            if self.test_loader is not None:
                avg_test_loss, test_accuracy = evaluate_model(model, criterion, self.test_loader)
                avg_training_loss = running_loss / total_train
                training_accuracy = 100 * correct_train / total_train
                print(f'Model {current_model_index}, '
                      f'Epoch [{epoch + 1}/{self.config["num_epochs"]}] '
                      f'Training Loss: {avg_training_loss:.4f}, Training Acc: {training_accuracy:.2f}%, '
                      f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

            # The below is required for verify_statistical_significance.py and data_vs_model_hardness_estimators.py
            if epoch + 1 == self.config['save_epoch']:
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
        return None

    def train_ensemble(
            self
    ):
        """Train an ensemble of models."""

        latest_model_indices = get_latest_model_index(self.save_dir, self.config['num_epochs'], self.dataset_count)

        print(f"Starting training {self.dataset_count} ensembles of {self.num_models_to_train_per_dataset} models each "
              f"on {self.dataset_name}.")
        print(f"Number of samples in the training loader: {len(cast(Sized, self.training_loaders[0].dataset))}")
        print(f"Number of samples in the test loader: {len(cast(Sized, self.test_loader.dataset))}")
        print('-'*20)

        for dataset_id in tqdm(range(self.dataset_count)):
            for model_id in tqdm(range(latest_model_indices[dataset_id] + 1, self.num_models_to_train_per_dataset)):
                hardness_estimates = {(dataset_id, model_id): {}}
                self.train_model(dataset_id, model_id, hardness_estimates)
                if self.estimate_hardness:
                    # Even though we computed multiple hardness estimates we only used AUM for our core experiments.
                    for estimator in ['Confidence', 'AUM', 'DataIQ', 'Loss']:
                        # Average hardness estimates (the ones that used learning dynamics) over all epochs.
                        hardness_estimates[(dataset_id, model_id)][estimator] = np.mean(
                            hardness_estimates[(dataset_id, model_id)][estimator], axis=1)
                    save_results(hardness_estimates, (dataset_id, model_id), self.dataset_name)
