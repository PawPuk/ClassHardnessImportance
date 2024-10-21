import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from neural_networks import ResNet18LowRes


class ModelTrainer:
    def __init__(self, training_loader, test_loader, num_epochs=200, lr=0.1, momentum=0.9, weight_decay=0.0005,
                 lr_decay_milestones=(60, 120, 160), save_epoch=20, save_dir='./Models/', dataset_type='full_',
                 dataset_name='MNIST', save_probe_models=True, timings_dir='./Timings/',
                 timings_file='ensemble_timings.csv'):
        """
        Initialize the ModelTrainer class.

        :param training_loader: DataLoader for the training dataset.
        :param test_loader: DataLoader for the test dataset.
        :param num_epochs: Number of epochs for training (default: 200).
        :param lr: Learning rate for the optimizer (default: 0.1).
        :param momentum: Momentum for the optimizer (default: 0.9).
        :param weight_decay: Weight decay for the optimizer (default: 0.0005).
        :param lr_decay_milestones: List of epochs at which to decay the learning rate (default: [60, 120, 160]).
        :param save_epoch: Epoch at which to optionally save the model (default: 20).
        :param save_dir: Base directory to save the models (default: './Models/').
        :param dataset_type: Type of dataset being used, e.g., 'full', 'fclp', 'dlp' (default: 'full').
        :param dataset_name: Name of the dataset being used (default: 'default_dataset').
        :param save_probe_models: Whether to save the probe models after save_epoch (default: True).
        :param timings_dir: Directory to save the timings (default: './Timings/').
        :param timings_file: File name to save ensemble timings (default: 'ensemble_timings.csv').
        """
        self.training_loader = training_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_decay_milestones = lr_decay_milestones
        self.save_epoch = save_epoch
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name  # New parameter for dataset name
        self.save_probe_models = save_probe_models

        # Incorporate dataset_name and dataset_type into directories to prevent overwriting
        self.save_dir = os.path.join(save_dir, dataset_type, dataset_name)
        self.timings_dir = os.path.join(timings_dir, dataset_type, dataset_name)
        self.timings_file = timings_file

        # Ensure directories exist
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.timings_dir, exist_ok=True)

    @staticmethod
    def create_model():
        """Creates and returns a ResNet-18 model."""
        return ResNet18LowRes(num_classes=10).cuda()

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

    def train_model(self, model_id):
        """Train a single model."""
        model = self.create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
                              nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_decay_milestones, gamma=0.2)

        for epoch in range(self.num_epochs):
            model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            for inputs, labels in self.training_loader:
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

            avg_train_loss = running_loss / total_train
            train_accuracy = 100 * correct_train / total_train

            # Evaluate the model on the test set
            avg_test_loss, test_accuracy = self.evaluate_model(model, criterion)

            print(f'Model {model_id}, Epoch [{epoch + 1}/{self.num_epochs}] '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                  f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

            # Optionally save the model at a specific epoch
            if self.save_probe_models and epoch + 1 == self.save_epoch:
                save_path = os.path.join(self.save_dir, f'_model_{model_id}_epoch_{self.save_epoch}.pth')
                torch.save(model.state_dict(), save_path)
                print(f'Model {model_id} ({self.dataset_name}, {self.dataset_type} dataset) saved at epoch '
                      f'{self.save_epoch}')

            # Step the scheduler at the end of the epoch
            scheduler.step()

        # Save model after full training
        final_save_path = os.path.join(self.save_dir, f'_model_{model_id}_epoch_{self.num_epochs}.pth')
        torch.save(model.state_dict(), final_save_path)
        print(f'Model {model_id} ({self.dataset_name}, {self.dataset_type} dataset) saved after full training at epoch '
              f'{self.num_epochs}')

    def train_ensemble(self, num_models=10):
        """Train an ensemble of models and measure the timing."""
        timings = []  # To store the training times for each model

        print(f"Starting training ensemble of {num_models} models on {self.dataset_name} ({self.dataset_type}) "
              f"dataset...")
        print(f"Number of samples in the training loader: {len(self.training_loader.dataset)}")
        print(f"Number of samples in the test loader: {len(self.test_loader.dataset)}")

        for model_id in tqdm(range(num_models)):
            start_time = time.time()  # Start time for training the model

            # Train the model
            self.train_model(model_id)

            # Calculate the time taken for training this model
            training_time = time.time() - start_time
            timings.append(training_time)
            print(f'Time taken for Model {model_id}: {training_time:.2f} seconds')

        # Calculate mean and standard deviation of the timings
        mean_time = np.mean(timings)
        std_time = np.std(timings)

        # Save the timings to a CSV file
        timings_file_path = os.path.join(self.timings_dir, self.timings_file)
        with open(timings_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write the header
            csvwriter.writerow(['Model ID', 'Training Time (seconds)'])
            # Write the timings for each model
            for model_id, timing in enumerate(timings):
                csvwriter.writerow([model_id, f'{timing:.2f}'])
            # Write the average and standard deviation
            csvwriter.writerow(['Average', f'{mean_time:.2f}'])
            csvwriter.writerow(['Std Dev', f'{std_time:.2f}'])

        print(f'Average time per model: {mean_time:.2f} seconds')
        print(f'Standard deviation of timings: {std_time:.2f} seconds')
