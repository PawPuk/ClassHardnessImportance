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
                 hardness='subjective', estimate_hardness=False, clean_data=False):
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
        self.estimate_hardness = estimate_hardness
        self.clean_data = 'clean' if clean_data else 'unclean'

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
        self.save_dir = str(os.path.join(config['save_dir'], pruning_type, f"{self.clean_data}{dataset_name}"))
        self.timings_dir = str(os.path.join(config['timings_dir'], pruning_type, f"{self.clean_data}{dataset_name}"))

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
            for inputs, labels, _ in self.test_loader:
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
        all_remembrance = [False for _ in range(self.total_samples)]
        all_forgetting = [0 for _ in range(self.total_samples)]

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

                if self.estimate_hardness:
                    index_within_batch = 0
                    for i, logit, label in zip(indices, outputs, labels):
                        i = i.item()
                        correct_logit = logit[label].item()
                        if label == predicted[index_within_batch]:
                            all_remembrance[i] = True
                        elif label != predicted[index_within_batch] and all_remembrance[i] == True:
                            all_remembrance[i] = False
                            all_forgetting[i] += 1
                        max_other_logit = torch.max(torch.cat((logit[:label], logit[label + 1:]))).item()
                        aum = correct_logit - max_other_logit

                        # Append the AUM for this sample
                        all_AUMs[i].append(aum)
                        index_within_batch += 1

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
        return all_AUMs, all_forgetting

    def save_results(self, all_AUMs, all_forgetting_statistics):
        """The purpose of this function is to enable easier generation of results. If we already spent a lot of
        resources on training an ensemble we don't want it to go to waste just because the ensemble is not large enough.
        We want to add more models to the ensemble rather than have to retrain it from scratch."""
        if self.estimate_hardness:
            hardness_save_dir = f"Results/{self.clean_data}{self.dataset_name}/"
            os.makedirs(hardness_save_dir, exist_ok=True)
            aum_path = os.path.join(hardness_save_dir, 'AUM.pkl')
            forgetting_path = os.path.join(hardness_save_dir, 'Forgetting.pkl')
            if os.path.exists(aum_path):
                with open(aum_path, "rb") as file:
                    existing_AUMs = pickle.load(file)
                for i in range(self.total_samples):
                    existing_AUMs[i].extend(all_AUMs[i])
                all_AUMs = existing_AUMs
            if os.path.exists(forgetting_path):
                with open(forgetting_path, "rb") as file:
                    existing_forgetting = pickle.load(file)
                for i in range(self.total_samples):
                    existing_forgetting[i] += all_forgetting_statistics[i]
                all_forgetting_statistics = existing_forgetting
            with open(aum_path, "wb") as file:
                pickle.dump(all_AUMs, file)
            with open(forgetting_path, "wb") as file:
                pickle.dump(all_forgetting_statistics, file)

    def train_ensemble(self):
        """Train an ensemble of models and measure the timing."""
        timings, all_AUMs, all_forgetting_statistics = [], [], []
        config = get_config(self.dataset_name)
        num_models = config['num_models']
        latest_model_index = self.get_latest_model_index()

        print(f"Starting training ensemble of {num_models} models on {self.dataset_name} ({self.pruning_type}) "
              f"dataset...")
        print(f"Number of samples in the training loader: {len(self.training_loader.dataset)}")
        print(f"Number of samples in the test loader: {len(self.test_loader.dataset)}")

        for model_id in tqdm(range(num_models)):
            start_time = time.time()
            AUMs, forgetting_statistics = self.train_model(model_id, latest_model_index + 1)
            all_AUMs.append(AUMs)
            all_forgetting_statistics.append(forgetting_statistics)
            training_time = time.time() - start_time
            timings.append((model_id + latest_model_index + 1, training_time))
            print(f'Time taken for Model {model_id + latest_model_index + 1}: {training_time:.2f} seconds')

        if self.estimate_hardness:
            self.save_results(all_AUMs, all_forgetting_statistics)

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
