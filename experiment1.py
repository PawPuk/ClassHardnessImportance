import torch
import torchvision
import torchvision.transforms as transforms
import os
from train_ensemble import ModelTrainer  # Import ModelTrainer from train_ensemble.py

# Define constants
BATCH_SIZE = 128

# Transformations for CIFAR-10 dataset
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 Dataset (Training set)
training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# CIFAR-10 Dataset (Test set)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Main script
if __name__ == '__main__':
    # Create an instance of ModelTrainer
    trainer = ModelTrainer(
        training_loader=training_loader,
        test_loader=test_loader,
        save_dir='./Models/FullDataset/',
        save_probe_models=True,
        timings_file='ensemble_timings.csv',
    )

    # Train the ensemble of models (10 models)
    trainer.train_ensemble(num_models=10)
