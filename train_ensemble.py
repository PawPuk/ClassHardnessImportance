import csv
import os
import pickle
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from neural_networks import ResNet18LowRes
from utils import get_config


class ModelTrainer:
    def __init__(self, training_loader, test_loader, dataset_name, pruning_type='none', save_probe_models=True,
                 hardness='subjective', compute_aum=False):
        """
        Initialize the ModelTrainer class with configuration specific to the dataset.

        :param training_loader: DataLoader for the training dataset.
        :param test_loader: DataLoader for the test dataset.
        :param dataset_name: Name of the dataset being used.
        :param pruning_type: Type of pruning being applied (default: 'none').
        :param save_probe_models: Whether to save the probe models after a specified epoch (default: True).
        :param hardness: Whether to use the same seed for measuring hardness (subjective) or different (objective) as
        for training the probe networks and benchmark ensemble (ensemble trained on unpruned data)
        """
        self.training_loader = training_loader
        self.test_loader = test_loader
        self.pruning_type = pruning_type
        self.dataset_name = dataset_name
        self.save_probe_models = save_probe_models

        # Fetch the dataset-specific configuration
        config = get_config(self.dataset_name)

        # Use the fetched config values
        self.num_epochs = config['num_epochs']
        self.lr = config['lr']
        self.momentum = config['momentum']
        self.weight_decay = config['weight_decay']
        self.lr_decay_milestones = config['lr_decay_milestones']
        self.save_epoch = config['save_epoch']
        self.num_classes = config['num_classes']
        self.total_samples = sum(config['num_training_samples'])
        if hardness == 'subjective':
            self.base_seed = config['probe_base_seed']
            self.seed_step = config['probe_seed_step']
        else:
            self.base_seed = config['new_base_seed']
            self.seed_step = config['new_seed_step']

        # Incorporate dataset_name and pruning_type into directories to prevent overwriting
        self.save_dir = str(os.path.join(config['save_dir'], pruning_type, dataset_name))
        self.timings_dir = str(os.path.join(config['timings_dir'], pruning_type, dataset_name))

        # Ensure directories exist
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.timings_dir, exist_ok=True)

    def get_latest_model_index(self):
        """Find the latest model index from saved files in the save directory."""
        max_index = -1
        if os.path.exists(self.save_dir):
            for filename in os.listdir(self.save_dir):
                match = re.search(r'model_(\d+)_epoch_200\.pth$', filename)
                if match:
                    index = int(match.group(1))
                    max_index = max(max_index, index)
        return max_index

    @staticmethod
    def set_seed(seed):
        """Set random seed for NumPy and PyTorch for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def create_model(self):
        """Creates and returns a ResNet-18 model with dynamic number of output classes."""
        return ResNet18LowRes(num_classes=self.num_classes).cuda()

    def evaluate_model(self, model, criterion):
        """Evaluate the model on the test set."""
        model.eval()
        correct, total, running_loss = 0, 0, 0.0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = running_loss / total
        return avg_loss, accuracy

    def train_model(self, model_id, current_model_index):
        """Train a single model."""
        seed = self.base_seed + (model_id + current_model_index) * self.seed_step
        self.set_seed(seed)

        model = self.create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
                              nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_decay_milestones, gamma=0.2)
        all_AUMs = [[] for _ in range(self.total_samples)]

        for epoch in range(self.num_epochs):
            model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            for inputs, labels, indices in self.training_loader:
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                for i, logit, label in zip(indices, outputs, labels):
                    i = i.item()
                    correct_logit = logit[label].item()
                    max_other_logit = torch.max(torch.cat((logit[:label], logit[label + 1:]))).item()
                    aum = correct_logit - max_other_logit

                    # Append the AUM for this sample
                    all_AUMs[i].append(aum)

            avg_train_loss = running_loss / total_train
            train_accuracy = 100 * correct_train / total_train

            # Evaluate the model on the test set
            avg_test_loss, test_accuracy = self.evaluate_model(model, criterion)

            print(f'Model {model_id + current_model_index}, Epoch [{epoch + 1}/{self.num_epochs}] '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                  f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

            # Optionally save the model at a specific epoch
            if self.save_probe_models and epoch + 1 == self.save_epoch:
                save_path = os.path.join(self.save_dir,
                                         f'model_{model_id + current_model_index}_epoch_{self.save_epoch}.pth')
                torch.save(model.state_dict(), save_path)
                print(f'Model {model_id + current_model_index} ({self.dataset_name}, {self.pruning_type} dataset) saved '
                      f'at epoch {self.save_epoch}')

            # Step the scheduler at the end of the epoch
            scheduler.step()

        # Save model after full training
        final_save_path = os.path.join(self.save_dir,
                                       f'model_{model_id + current_model_index}_epoch_{self.num_epochs}.pth')
        torch.save(model.state_dict(), final_save_path)
        print(f'Model {model_id + current_model_index} ({self.dataset_name}, {self.pruning_type} dataset) saved after '
              f'full training at epoch {self.num_epochs}')
        return all_AUMs

    def train_ensemble(self):
        """Train an ensemble of models and measure the timing."""
        timings = []  # To store the training times for each model
        config = get_config(self.dataset_name)
        num_models = config['num_models']

        print(f"Starting training ensemble of {num_models} models on {self.dataset_name} ({self.pruning_type}) "
              f"dataset...")
        print(f"Number of samples in the training loader: {len(self.training_loader.dataset)}")
        print(f"Number of samples in the test loader: {len(self.test_loader.dataset)}")

        latest_model_index = self.get_latest_model_index()

        all_AUMs = []

        for model_id in tqdm(range(num_models)):
            start_time = time.time()  # Start time for training the model

            # Train the model
            all_AUMs.append(self.train_model(model_id, latest_model_index + 1))

            # Calculate the time taken for training this model
            training_time = time.time() - start_time
            timings.append((model_id + latest_model_index + 1, training_time))
            print(f'Time taken for Model {model_id + latest_model_index + 1}: {training_time:.2f} seconds')

        averaged_AUMs = [
            [
                sum(model_list[sample_idx][epoch_idx] for model_list in all_AUMs) / len(all_AUMs)
                for epoch_idx in range(self.num_epochs)
            ]
            for sample_idx in range(self.total_samples)
        ]

        AUM_save_dir = f'Results/{self.dataset_name}/AUM.pkl'
        os.makedirs(AUM_save_dir, exist_ok=True)
        with open(AUM_save_dir, "wb") as file:
            pickle.dump(averaged_AUMs, file)

        # Calculate mean and standard deviation of the timings
        timing_values = [timing[1] for timing in timings]
        mean_time = np.mean(timing_values)
        std_time = np.std(timing_values)

        # Save the timings to a CSV file
        timings_file_path = os.path.join(self.timings_dir, 'ensemble_timings.csv')
        file_exists = os.path.exists(timings_file_path)
        with open(timings_file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write the header
            if not file_exists:
                csvwriter.writerow(['Model ID', 'Training Time (seconds)'])
            # Write the timings for each model
            for model_id, timing in timings:
                csvwriter.writerow([model_id, f'{timing:.2f}'])
            # Write the average and standard deviation
            csvwriter.writerow(['Average', f'{mean_time:.2f}'])
            csvwriter.writerow(['Std Dev', f'{std_time:.2f}'])

        print(f'Average time per model: {mean_time:.2f} seconds')
        print(f'Standard deviation of timings: {std_time:.2f} seconds')
