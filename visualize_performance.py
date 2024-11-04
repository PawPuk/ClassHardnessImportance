import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from neural_networks import ResNet18LowRes


def load_models(pruning_strategy: str, dataset_name: str) -> Dict[int, List[dict]]:
    """
    Load all models for the specified pruning strategy and dataset.

    :param pruning_strategy: Abbreviated pruning strategy (e.g., 'loop' for leave_one_out_pruning).
    :param dataset_name: Name of the dataset (e.g., 'CIFAR10').
    :return: Dictionary where keys are pruning rates and values are lists of model state dictionaries.
    """
    models_dir = "Models"
    models_by_rate = {}

    # Walk through each folder in the Models directory
    for root, dirs, files in os.walk(models_dir):
        # Check if the path matches the specified pruning strategy and dataset name
        if f"{pruning_strategy}" in root and f"{dataset_name}" in root:
            pruning_rate = int(root.split(pruning_strategy)[1].split("/")[0])
            models_by_rate.setdefault(pruning_rate, [])
            for file in files:
                if file.endswith(".pth") and "_epoch_200" in file:
                    model_path = os.path.join(root, file)
                    model_state = torch.load(model_path)
                    models_by_rate[pruning_rate].append(model_state)
                    print(f"Loaded model for pruning rate {pruning_rate}: {model_path}")

    # Load models trained on the full dataset (no pruning)
    full_dataset_dir = os.path.join(models_dir, "none", dataset_name)
    if os.path.exists(full_dataset_dir):
        models_by_rate[0] = []  # Use `0` to represent models without pruning
        for file in os.listdir(full_dataset_dir):
            if file.endswith(".pth") and "_epoch_200" in file:
                model_path = os.path.join(full_dataset_dir, file)
                model_state = torch.load(model_path)
                models_by_rate[0].append(model_state)
                print(f"Loaded model for full dataset (no pruning): {model_path}")


    print(f"Models loaded by pruning rate for {pruning_strategy} on {dataset_name}")
    return models_by_rate


def load_cifar10_test_set(batch_size: int = 64) -> DataLoader:
    """
    Load CIFAR-10 test set.

    :param batch_size: Batch size for the test DataLoader. Default is 64.
    :return: DataLoader for the CIFAR-10 test set.
    """
    # Define the transformation for the test set (only normalization here)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load the CIFAR-10 test set
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return test_loader


def evaluate_ensemble(ensemble: List[dict], test_loader) -> List[float]:
    """
    Evaluate an ensemble of models using logit averaging.

    :param ensemble: List of model state dictionaries (one per model in the ensemble).
    :param test_loader: DataLoader for the CIFAR-10 test set.
    :return: List of 11 accuracies - one for the entire dataset and ten for each class.
    """
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize lists to store logits for each model's predictions
    all_logits = []

    # Iterate through the ensemble, load models, and compute logits
    for model_state in ensemble:
        # Load the model
        model = ResNet18LowRes(10)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()

        # Store logits for the current model
        model_logits = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                model_logits.append(outputs.cpu())

        # Stack all batches' logits for this model
        all_logits.append(torch.cat(model_logits, dim=0))

    # Compute the average logits across the ensemble
    avg_logits = torch.stack(all_logits).mean(dim=0)

    # Compute dataset-level accuracy and per-class accuracy
    dataset_correct = 0
    class_correct = torch.zeros(10)
    class_counts = torch.zeros(10)

    # Generate predictions from averaged logits
    _, predicted = avg_logits.max(1)

    # Iterate over the test set to compute accuracy per class
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            batch_start = i * test_loader.batch_size
            batch_end = batch_start + labels.size(0)
            batch_predictions = predicted[batch_start:batch_end]

            # Update dataset-level accuracy count
            dataset_correct += (batch_predictions == labels).sum().item()

            # Update class-level accuracy counts
            for class_id in range(10):
                class_mask = (labels == class_id)
                class_correct[class_id] += (batch_predictions[class_mask] == class_id).sum().item()
                class_counts[class_id] += class_mask.sum().item()

    # Calculate overall accuracy and per-class accuracies
    dataset_accuracy = 100.0 * dataset_correct / len(test_loader.dataset)
    class_accuracies = [100.0 * class_correct[i] / class_counts[i] for i in range(10)]

    return [dataset_accuracy] + class_accuracies


def evaluate_individual_models(ensemble: List[dict], test_loader) -> List[float]:
    """
    Evaluate each model individually in the ensemble and return the average accuracy.

    :param ensemble: List of model state dictionaries (one per model in the ensemble).
    :param test_loader: DataLoader for the CIFAR-10 test set.
    :return: List of 11 accuracies - one for the entire dataset and ten for each class, averaged across all models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize accuracy accumulators
    dataset_accuracies, class_accuracies = [], []

    for model_state in ensemble:
        model = ResNet18LowRes(10)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()

        # Initialize counters
        dataset_correct = 0
        class_correct = torch.zeros(10)
        class_counts = torch.zeros(10)

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)

                # Dataset-level accuracy
                dataset_correct += (predicted == labels).sum().item()

                # Class-level accuracy
                for class_id in range(10):
                    class_mask = (labels == class_id)
                    class_correct[class_id] += (predicted[class_mask] == class_id).sum().item()
                    class_counts[class_id] += class_mask.sum().item()

        # Compute accuracy for this model
        dataset_accuracy = 100.0 * dataset_correct / len(test_loader.dataset)
        per_class_accuracies = [100.0 * class_correct[i] / class_counts[i] for i in range(10)]

        # Append to accuracy lists
        dataset_accuracies.append(dataset_accuracy)
        class_accuracies.append(per_class_accuracies)

    # Average dataset accuracy and per-class accuracies across all models
    avg_dataset_accuracy = sum(dataset_accuracies) / len(dataset_accuracies)
    avg_class_accuracies = [sum(acc[i] for acc in class_accuracies) / len(class_accuracies) for i in range(10)]

    return [avg_dataset_accuracy] + avg_class_accuracies


def evaluate_all_ensembles(models_by_rate: Dict[int, List[dict]], test_loader) -> Dict[int, List[float]]:
    """
    Evaluate all ensembles and return accuracies per pruning rate.

    :param models_by_rate: Dictionary where keys are pruning rates and values are lists of model state dicts.
    :param test_loader: DataLoader for the CIFAR-10 test set.
    :return: Dictionary with pruning rates as keys and list of 11 accuracy values as values.
    """
    ensemble_accuracies = {}

    for pruning_rate, ensemble in models_by_rate.items():
        print(f"Evaluating ensemble for pruning rate {pruning_rate}%...")
        ensemble_accuracies[pruning_rate] = evaluate_ensemble(ensemble, test_loader)

    return ensemble_accuracies


def evaluate_all_models_individually(models_by_rate: Dict[int, List[dict]], test_loader) -> Dict[int, List[float]]:
    """
    Evaluate individual model performance across pruning rates.

    :param models_by_rate: Dictionary where keys are pruning rates and values are lists of model state dicts.
    :param test_loader: DataLoader for the CIFAR-10 test set.
    :return: Dictionary with pruning rates as keys and list of 11 accuracy values as values.
    """
    individual_model_accuracies = {}

    for pruning_rate, ensemble in models_by_rate.items():
        print(f"Evaluating individual models for pruning rate {pruning_rate}%...")
        individual_model_accuracies[pruning_rate] = evaluate_individual_models(ensemble, test_loader)

    return individual_model_accuracies


def plot_ensemble_accuracies(ensemble_accuracies: Dict[int, List[float]]):
    """
    Plot class-level and dataset-level accuracies across pruning rates.

    :param ensemble_accuracies: Dictionary with pruning rates as keys and list of 11 accuracy values as values.
                                The first value in each list is the dataset-level accuracy, and the rest are
                                class-level accuracies.
    """
    # Sort the pruning rates for consistent plotting
    pruning_rates = sorted(ensemble_accuracies.keys())

    # Extract dataset and class accuracies across all pruning rates
    dataset_accuracies = [ensemble_accuracies[rate][0] for rate in pruning_rates]
    class_accuracies = [[ensemble_accuracies[rate][i + 1] for rate in pruning_rates] for i in range(10)]

    # Create a figure with 10 subplots for each class
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey=True)
    fig.suptitle('Class-Level and Dataset-Level Accuracy Across Pruning Rates', fontsize=16)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Plot class-level and dataset-level accuracy for each class
    for class_id in range(10):
        ax = axes[class_id]
        ax.plot(pruning_rates, class_accuracies[class_id], label=f'Class {class_id} Accuracy', marker='o')
        ax.plot(pruning_rates, dataset_accuracies, label='Dataset Accuracy', linestyle='--', marker='x')

        # Customize each subplot
        ax.set_title(f'Class {class_id} Accuracy')
        ax.set_xlabel('Pruning Rate (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    plt.savefig('Figure.pdf')


def plot_individual_model_accuracies(individual_model_accuracies: Dict[int, List[float]]):
    """
    Plot class-level and dataset-level accuracies across pruning rates for individual models.

    :param individual_model_accuracies: Dictionary with pruning rates as keys and list of 11 accuracy values as values.
                                        The first value in each list is the dataset-level accuracy, and the rest are
                                        class-level accuracies.
    """
    pruning_rates = sorted(individual_model_accuracies.keys())

    dataset_accuracies = [individual_model_accuracies[rate][0] for rate in pruning_rates]
    class_accuracies = [[individual_model_accuracies[rate][i + 1] for rate in pruning_rates] for i in range(10)]

    fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey=True)
    fig.suptitle('Individual Model Class-Level and Dataset-Level Accuracy Across Pruning Rates', fontsize=16)

    axes = axes.flatten()

    for class_id in range(10):
        ax = axes[class_id]
        ax.plot(pruning_rates, class_accuracies[class_id], label=f'Class {class_id} Accuracy', marker='o')
        ax.plot(pruning_rates, dataset_accuracies, label='Dataset Accuracy', linestyle='--', marker='x')

        ax.set_title(f'Class {class_id} Accuracy')
        ax.set_xlabel('Pruning Rate (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('Figure1.pdf')
    plt.close()


def main(pruning_strategy, dataset_name):
    models = load_models(pruning_strategy, dataset_name)
    test_loader = load_cifar10_test_set(256)
    # Evaluate ensemble performance
    ensemble_results = evaluate_all_ensembles(models, test_loader)
    plot_ensemble_accuracies(ensemble_results)
    # Evaluate individual models' performance
    individual_model_results = evaluate_all_models_individually(models, test_loader)
    plot_individual_model_accuracies(individual_model_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--pruning_strategy", type=str, required=True,
                        help="Abbreviated pruning strategy (e.g., 'loop' for leave_one_out_pruning)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (e.g., 'CIFAR10')")
    args = parser.parse_args()
    main(args.pruning_strategy, args.dataset_name)
