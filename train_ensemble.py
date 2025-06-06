import csv
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import get_config, ROOT
from neural_networks import ResNet18LowRes
from utils import set_reproducibility, get_latest_model_index


class ModelTrainer:
    def __init__(self, training_set_size, training_loader, test_loader, dataset_name, pruning_type='none',
                 save_probe_models=True, estimate_hardness=False, clean_data=False):
        """
        Initialize the ModelTrainer class with configuration specific to the dataset.

        :param training_set_size: Specified the size of the training set
        :param training_loader: DataLoader for the training dataset.
        :param test_loader: DataLoader for the test dataset.
        :param dataset_name: The name of the dataset being used.
        :param pruning_type: Type of pruning being applied (default: 'none').
        :param save_probe_models: Whether to save the probe models after a specified epoch (default: True).
        :param estimate_hardness: Specifies if the hardness should be saved and stored during training (default False).
        :param clean_data: Allows differentiating between models trained on clean and noisy data while saving.
        """
        self.training_set_size = training_set_size
        self.training_loader = training_loader
        self.test_loader = test_loader
        self.pruning_type = pruning_type
        self.dataset_name = dataset_name
        self.save_probe_models = save_probe_models
        self.estimate_hardness = estimate_hardness
        self.clean_data = 'clean' if clean_data else 'unclean'

        self.config = get_config(self.dataset_name)

        # Incorporate dataset_name and pruning_type into directories to prevent overwriting
        self.save_dir = str(os.path.join(self.config['save_dir'], pruning_type, f"{self.clean_data}{dataset_name}"))
        self.timings_dir = str(os.path.join(self.config['timings_dir'], pruning_type,
                                            f"{self.clean_data}{dataset_name}"))
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.timings_dir, exist_ok=True)

    def compute_current_seed(self, current_model_index):
        base_seed = self.config['probe_base_seed']
        seed_step = self.config['probe_seed_step']
        seed = base_seed + current_model_index * seed_step
        return seed

    @staticmethod
    def estimate_instance_hardness(indices, outputs, labels, predicted, all_AUMs, all_remembrance, all_forgetting):
        for index_within_batch, (i, logit, label) in enumerate(zip(indices, outputs, labels)):
            i = i.item()  # TODO: Maybe change the output of __iter__ rather than doing the .item() here.

            # Compute Forgetting events
            if label == predicted[index_within_batch]:
                all_remembrance[i] = True
            elif label != predicted[index_within_batch] and all_remembrance[i] == True:
                all_remembrance[i] = False
                all_forgetting[i] += 1

            # Compute AUM values
            correct_logit = logit[label].item()
            max_other_logit = torch.max(torch.cat((logit[:label], logit[label + 1:]))).item()
            aum = correct_logit - max_other_logit
            all_AUMs[i].append(aum)

    def evaluate_model(self, model, criterion):
        """Evaluate the model on the test set."""
        model.eval()
        correct, total, running_loss = 0, 0, 0.0

        with torch.no_grad():
            for inputs, labels, _ in self.test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)  # TODO: Why * inputs.size(0) ???????
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = running_loss / total
        return avg_loss, accuracy

    def train_model(self, current_model_index):
        """Train a single model."""
        seed = self.compute_current_seed(current_model_index)
        set_reproducibility(seed)

        model = ResNet18LowRes(num_classes=self.config['num_classes']).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'],
                              weight_decay=self.config['weight_decay'], nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config['lr_decay_milestones'], gamma=0.2)

        all_AUMs = [[] for _ in range(self.training_set_size)]  # all_AUMs[sample_index][epoch_index]
        all_remembrance = [False for _ in range(self.training_set_size)]
        all_forgetting = [0 for _ in range(self.training_set_size)]

        for epoch in range(self.config['num_epochs']):
            model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            for inputs, labels, indices in self.training_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)  # TODO: Why * inputs.size(0) ???????
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                if self.estimate_hardness:
                    self.estimate_instance_hardness(indices, outputs, labels, predicted, all_AUMs, all_remembrance,
                                                    all_forgetting)
            scheduler.step()

            avg_train_loss = running_loss / total_train
            train_accuracy = 100 * correct_train / total_train
            avg_test_loss, test_accuracy = self.evaluate_model(model, criterion)
            print(f'Model {current_model_index}, Epoch [{epoch + 1}/{self.config["num_epochs"]}] '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                  f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

            # Optionally save the model at a specific epoch
            if self.save_probe_models and epoch + 1 == self.config['save_epoch']:
                save_path = os.path.join(self.save_dir, f'model_{current_model_index}_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), save_path)
                print(f'Model {current_model_index} ({self.dataset_name}, {self.pruning_type} dataset) saved'
                      f' at epoch {self.config["save_epoch"]}.')

        # Save model after full training
        final_save_path = os.path.join(self.save_dir,
                                       f'model_{current_model_index}_epoch_{self.config["num_epochs"]}.pth')
        torch.save(model.state_dict(), final_save_path)
        print(f'Model {current_model_index} ({self.dataset_name}, {self.pruning_type} dataset) saved after '
              f'full training at epoch {self.config["num_epochs"]}.')
        return all_AUMs, all_forgetting

    @staticmethod
    def load_previous_hardness_estimates(path):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as file:
                prior_hardness_estimates = pickle.load(file)
            print(f'{path} exists - extended hardness estimates.')
            return prior_hardness_estimates
        else:
            print(f"{path} does not exist or is empty. Initializing new data.")
            return []

    def save_results(self, all_AUMs, all_forgetting_statistics):
        """
        The purpose of this function is to enable easier generation of results. If we already spent a lot of
        resources on training an ensemble, we don't want it to go to waste just because the ensemble is not large
        enough. We want to add more models to the ensemble rather than have to retrain it from scratch.
        """
        hardness_save_dir = os.path.join(ROOT, f"Results/{self.clean_data}{self.dataset_name}/")
        os.makedirs(hardness_save_dir, exist_ok=True)
        aum_path = os.path.join(hardness_save_dir, 'AUM.pkl')
        forgetting_path = os.path.join(hardness_save_dir, 'Forgetting.pkl')

        # Save updated AUM.pkl
        all_AUMs = all_AUMs + self.load_previous_hardness_estimates(aum_path)
        with open(aum_path, "wb") as file:
            print(f'Saving AUMs of the following shape: {len(all_AUMs)}, {len(all_AUMs[0])}, {len(all_AUMs[0][0])}.')
            pickle.dump(all_AUMs, file)

        # Save updated Forgetting.pkl
        all_forgetting_statistics = all_forgetting_statistics + self.load_previous_hardness_estimates(forgetting_path)
        with open(forgetting_path, "wb") as file:
            print(f'Saving forgettings of the following shape: '
                  f'{len(all_forgetting_statistics)}, {len(all_forgetting_statistics[0])}.')
            pickle.dump(all_forgetting_statistics, file)

    def train_ensemble(self):
        """Train an ensemble of models and measure the timing."""
        timings, all_AUMs, all_forgetting_statistics = [], [], []
        latest_model_index = get_latest_model_index(self.save_dir, self.config['num_epochs'])
        num_models = self.config['robust_ensemble_size'] if \
            self.config['robust_ensemble_size'] < self.config['num_models'] else self.config['num_models']

        print(f"Starting training ensemble of {num_models} models on {self.dataset_name}.")
        print(f"Number of samples in the training loader: {len(self.training_loader.dataset)}")
        print(f"Number of samples in the test loader: {len(self.test_loader.dataset)}")
        print('-'*20)

        for model_id in tqdm(range(num_models)):
            start_time = time.time()
            AUMs, forgetting_statistics = self.train_model(model_id + latest_model_index + 1)
            all_AUMs.append(AUMs)
            all_forgetting_statistics.append(forgetting_statistics)
            training_time = time.time() - start_time
            timings.append((model_id + latest_model_index + 1, training_time))

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
