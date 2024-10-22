import argparse
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt

from data_pruning import DataPruning
from neural_networks import ResNet18LowRes
from train_ensemble import ModelTrainer

# Define constants
BATCH_SIZE = 128
SAVE_EPOCH = 20  # Models saved after 20 epochs
MODEL_DIR = './Models/'
NUM_CLASSES = 10

# Transformations for CIFAR-10 dataset (Training and Test sets)
# Training set transformation includes data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Test set transformation: no augmentation, only normalization
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 Dataset (Training set)
training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
TRAINING_SET_SIZE = len(training_set)

# CIFAR-10 Dataset (Test set)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# Define the ResNet-18 model
def create_model():
    model = ResNet18LowRes(num_classes=NUM_CLASSES)
    return model


# Function to calculate EL2N score for each sample
def compute_el2n(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    el2n_scores = []  # Store EL2N scores for all training samples

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            # Apply softmax to model outputs
            softmax_outputs = F.softmax(outputs, dim=1)

            # One-hot encode the labels
            one_hot_labels = F.one_hot(labels, num_classes=NUM_CLASSES).float()

            # Compute the L2 norm of the error
            l2_errors = torch.norm(softmax_outputs - one_hot_labels, dim=1)  # L2 norm along the class dimension

            # Extend the list with L2 errors for this batch
            el2n_scores.extend(l2_errors.cpu().numpy())  # Convert to CPU and add to the list

    return el2n_scores


# Function to load a saved model and return the EL2N scores
def load_model_and_compute_el2n(model_id, dataset_name):
    model = create_model().cuda()
    # Construct the model path based on pruning strategy, dataset name, and save epoch
    model_path = os.path.join(MODEL_DIR, 'full', dataset_name, f'model_{model_id}_epoch_{SAVE_EPOCH}.pth')

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f'Model {model_id} loaded successfully from epoch {SAVE_EPOCH}.')

        # Compute EL2N scores for this model on the training set
        el2n_scores = compute_el2n(model, training_loader)
        return el2n_scores
    else:
        print(f'Model {model_id} not found at epoch {SAVE_EPOCH}.')
        return None


# Function to aggregate EL2N scores across all models
def collect_el2n_scores(dataset_name):
    all_el2n_scores = [[] for _ in range(TRAINING_SET_SIZE)]  # List of lists to store scores from each model

    # Loop over all 10 models
    for model_id in range(10):
        el2n_scores = load_model_and_compute_el2n(model_id, dataset_name)
        if el2n_scores:
            # Store EL2N scores from this model into the master list
            for i in range(TRAINING_SET_SIZE):
                all_el2n_scores[i].append(el2n_scores[i])

    return all_el2n_scores


# Function to group EL2N scores by class
def group_scores_by_class(el2n_scores, training_set):
    class_el2n_scores = {i: [] for i in range(NUM_CLASSES)}  # Dictionary to store scores by class
    labels = []  # Store corresponding labels

    # Since we are not shuffling the data loader, we can directly match scores with their labels
    for i, (_, label) in enumerate(training_set):
        class_el2n_scores[label].append(el2n_scores[i])
        labels.append(label)  # Collect the labels

    return class_el2n_scores, labels


# Function to compute statistics for candlestick chart
def compute_class_statistics(class_el2n_scores):
    class_stats = {}

    for class_id, scores in class_el2n_scores.items():
        scores_array = np.array(scores)  # Convert to numpy array for statistical analysis

        # Compute mean, std, and quartiles
        means = np.mean(scores_array, axis=1)
        q1 = np.percentile(means, 25)
        q3 = np.percentile(means, 75)
        min_val = np.min(means)
        max_val = np.max(means)

        class_stats[class_id] = {
            "q1": q1,
            "q3": q3,
            "min": min_val,
            "max": max_val
        }

    return class_stats


# Function to plot candlestick chart for class-level EL2N scores
def plot_class_level_candlestick(class_stats, dataset_name, pruning_strategy):
    # Prepare the saving directory and file name
    save_dir = os.path.join('Figures/', pruning_strategy, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f'{pruning_strategy}{dataset_name}_hardness_distribution.pdf')

    # Prepare the data for plotting
    class_ids = list(class_stats.keys())
    q1_values = [class_stats[class_id]["q1"] for class_id in class_ids]
    q3_values = [class_stats[class_id]["q3"] for class_id in class_ids]
    min_values = [class_stats[class_id]["min"] for class_id in class_ids]
    max_values = [class_stats[class_id]["max"] for class_id in class_ids]

    # Create the candlestick chart
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(NUM_CLASSES):
        # Draw the candlestick (real body: Q1 to Q3, shadow: min to max)
        ax.plot([i, i], [min_values[i], max_values[i]], color='black')  # Shadow
        ax.plot([i, i], [q1_values[i], q3_values[i]], color='blue', lw=6)  # Real body

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([f'Class {i}' for i in range(NUM_CLASSES)])
    ax.set_xlabel("Classes")
    ax.set_ylabel("EL2N Score (L2 Norm)")
    ax.set_title("Class-Level EL2N Scores Candlestick Plot")
    plt.savefig(file_name)


# Function to create a pruned dataset
def prune_dataset(el2n_scores, class_el2n_scores, labels, pruning_strategy, pruning_rate, dataset_name):
    # Instantiate the DataPruning class with el2n_scores, class_el2n_scores, and labels
    pruner = DataPruning(el2n_scores, class_el2n_scores, labels, pruning_rate, dataset_name)

    if pruning_strategy == 'dlp':
        pruned_indices = pruner.dataset_level_pruning()
    elif pruning_strategy == 'fclp':
        pruned_indices = pruner.fixed_class_level_pruning()
    else:
        raise ValueError('Wrong value of the parameter `pruning_strategy`.')

    # Create a new pruned dataset
    pruned_dataset = torch.utils.data.Subset(training_set, pruned_indices)

    return pruned_dataset


# Main script
def main(pruning_strategy, dataset_name, pruning_rate):

    # Collect EL2N scores across all models for the training set
    all_el2n_scores = collect_el2n_scores(dataset_name)

    # Group EL2N scores by class and get labels
    class_el2n_scores, labels = group_scores_by_class(all_el2n_scores, training_set)

    # Compute class-level statistics for candlestick chart
    class_stats = compute_class_statistics(class_el2n_scores)

    # Plot the class-level candlestick chart
    plot_class_level_candlestick(class_stats, dataset_name, pruning_strategy)

    # Perform dataset-level pruning
    pruned_dataset = prune_dataset(all_el2n_scores, class_el2n_scores, labels, pruning_strategy, pruning_rate,
                                   dataset_name)

    # Create data loader for pruned dataset
    pruned_training_loader = torch.utils.data.DataLoader(pruned_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                         num_workers=2)

    # Train ensemble of 10 models on pruned data (without saving probe models)
    trainer = ModelTrainer(pruned_training_loader, test_loader, dataset_name, f'{pruning_strategy + str(pruning_rate)}',
                           False)
    trainer.train_ensemble()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EL2N Score Calculation and Dataset Pruning')
    parser.add_argument('--pruning_strategy', type=str, default='dlp', choices=['fclp', 'dlp'],
                        help='Choose pruning strategy: fclp (fixed class level pruning) or dlp (data level pruning)')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='Specify the dataset name (default: CIFAR10)')
    parser.add_argument('--pruning_rate', type=int, default=50,
                        help='Percentage of data samples that will be removed during data pruning (use integers).')

    args = parser.parse_args()

    main(args.pruning_strategy, args.dataset_name, args.pruning_rate)