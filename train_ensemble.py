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
        self.num_epochs = self.config['num_epochs']
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
    def estimate_instance_hardness(batch_indices, inputs, outputs, labels, predicted, hardness_estimates, epoch,
                                   remembering):
        for index_within_batch, (i, x, logits, label) in enumerate(zip(batch_indices, inputs, outputs, labels)):
            i = i.item()  # TODO: Maybe change the output of __iter__ rather than doing the .item() here.
            correct_label = label.item()
            predicted_label = predicted[index_within_batch].item()

            logits = logits.detach()
            correct_logit = logits[correct_label].item()
            probs = torch.nn.functional.softmax(logits, dim=0)
            # Confidence
            hardness_estimates['Confidence'][i][epoch] = correct_logit
            # AUM
            max_other_logit = torch.max(torch.cat((logits[:correct_label], logits[correct_label + 1:]))).item()
            hardness_estimates['AUM'][i][epoch] = correct_logit - max_other_logit
            # DataIQ
            p_y = probs[correct_label].item()
            hardness_estimates['DataIQ'][i][epoch] = p_y * (1 - p_y)
            # Cross-Entropy Loss
            label_tensor = torch.tensor([correct_label], device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), label_tensor).item()
            hardness_estimates['Loss'][i][epoch] = loss
            # Forgetting
            if predicted_label == correct_label:
                remembering[i] = True
            elif predicted_label != correct_label and remembering[i]:
                hardness_estimates['Forgetting'][i] += 1
                remembering[i] = False

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

    def train_model(self, current_model_index, latest_model_index, hardness_estimates):
        """Train a single model."""
        seed = self.compute_current_seed(current_model_index + latest_model_index + 1)
        set_reproducibility(seed)

        model = ResNet18LowRes(num_classes=self.config['num_classes']).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'],
                              weight_decay=self.config['weight_decay'], nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config['lr_decay_milestones'], gamma=0.2)

        for estimator in ['Confidence', 'AUM', 'DataIQ', 'Loss']:   # [epoch_index][sample_index]
            hardness_estimates[estimator] = [[None for _ in range(self.num_epochs)]
                                             for _ in range(self.training_set_size)]
        hardness_estimates['Forgetting'] = [0 for _ in range(self.training_set_size)]
        remembering = [False for _ in range(self.training_set_size)]

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
                    self.estimate_instance_hardness(indices, inputs, outputs, labels, predicted, hardness_estimates,
                                                    epoch, remembering)
            scheduler.step()

            avg_train_loss = running_loss / total_train
            train_accuracy = 100 * correct_train / total_train
            avg_test_loss, test_accuracy = self.evaluate_model(model, criterion)
            print(f'Model {current_model_index + latest_model_index + 1}, '
                  f'Epoch [{epoch + 1}/{self.config["num_epochs"]}] '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                  f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

            # Optionally save the model at a specific epoch
            if self.save_probe_models and epoch + 1 == self.config['save_epoch']:
                save_path = os.path.join(self.save_dir,
                                         f'model_{current_model_index + latest_model_index + 1}_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), save_path)
                print(f'Model {current_model_index + latest_model_index + 1} ({self.dataset_name}, '
                      f'{self.pruning_type} dataset) saved at epoch {self.config["save_epoch"]}.')

        # Save model after full training
        final_save_path = os.path.join(self.save_dir,
                                       f'model_{current_model_index + latest_model_index + 1}_epoch_'
                                       f'{self.config["num_epochs"]}.pth')
        torch.save(model.state_dict(), final_save_path)
        print(f'Model {current_model_index + latest_model_index + 1} ({self.dataset_name}, {self.pruning_type} dataset) '
              f'saved after full training at epoch {self.config["num_epochs"]}.')

    @staticmethod
    def load_previous_hardness_estimates(path):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as file:
                prior_hardness_estimates = pickle.load(file)
            print(f'{path} exists - extended hardness estimates.')
            return prior_hardness_estimates
        else:
            print(f"{path} does not exist or is empty. Initializing new data.")
            return {'Confidence': [], 'AUM': [], 'DataIQ': [], 'Forgetting': [], 'Loss': []}

    def save_results(self, hardness_estimates):
        """
        The purpose of this function is to enable easier generation of results. If we already spent a lot of
        resources on training an ensemble, we don't want it to go to waste just because the ensemble is not large
        enough. We want to add more models to the ensemble rather than have to retrain it from scratch.
        """
        hardness_save_dir = os.path.join(ROOT, f"Results/{self.clean_data}{self.dataset_name}/")
        os.makedirs(hardness_save_dir, exist_ok=True)
        path = os.path.join(hardness_save_dir, 'hardness_estimates.pkl')
        old_hardness_estimates = self.load_previous_hardness_estimates(path)

        for estimator in hardness_estimates.keys():
            old_hardness_estimates[estimator].append(hardness_estimates[estimator])

        with open(path, "wb") as file:
            print(f'Saving updated hardness estimates.')
            pickle.dump(old_hardness_estimates, file)

    def train_ensemble(self):
        """Train an ensemble of models and measure the timing."""
        timings = []

        latest_model_index = get_latest_model_index(self.save_dir, self.config['num_epochs'])
        num_models = self.config['robust_ensemble_size'] if \
            self.config['robust_ensemble_size'] < self.config['num_models'] else self.config['num_models']

        print(f"Starting training ensemble of {num_models} models on {self.dataset_name}.")
        print(f"Number of samples in the training loader: {len(self.training_loader.dataset)}")
        print(f"Number of samples in the test loader: {len(self.test_loader.dataset)}")
        print('-'*20)

        for model_id in tqdm(range(num_models)):
            hardness_estimates = {}
            start_time = time.time()
            self.train_model(model_id, latest_model_index, hardness_estimates)
            training_time = time.time() - start_time
            timings.append((model_id + latest_model_index + 1, training_time))

            if self.estimate_hardness:
                for estimator in ['Confidence', 'AUM', 'DataIQ', 'Loss']:
                    hardness_estimates[estimator] = np.mean(hardness_estimates[estimator], axis=1)
                self.save_results(hardness_estimates)

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
