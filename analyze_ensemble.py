import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neural_networks import ResNet18LowRes
from utils import get_config


def load_model(model_path, num_classes):
    """Loads a single model from a specified path."""
    model = ResNet18LowRes(num_classes=num_classes).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_test_loader(config, dataset_name):
    """
    Loads and returns the test DataLoader for the specified dataset.

    :param config: Name of the dataset for which to load the test data.
    :param dataset_name: Name of the dataset to load.
    :return: DataLoader for the test dataset.
    """
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std'])
    ])

    if dataset_name == 'CIFAR10':
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    elif dataset_name == 'CIFAR100':
        test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported. Choose either CIFAR10 or CIFAR100.")

    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    return test_loader


def ensemble_accuracies(models, test_loader, num_classes):
    """Computes the dataset-level and class-level accuracies for an ensemble of models."""
    correct_class_counts = np.zeros(num_classes)
    total_class_counts = np.zeros(num_classes)
    dataset_correct, dataset_total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            # Use logit averaging for measuring accuracies
            outputs_sum = sum(model(inputs) for model in models)
            _, predicted = torch.max(outputs_sum, 1)

            dataset_correct += (predicted == labels).sum().item()
            dataset_total += labels.size(0)

            for c in range(num_classes):
                correct_class_counts[c] += ((predicted == labels) & (labels == c)).sum().item()
                total_class_counts[c] += (labels == c).sum().item()

    dataset_accuracy = 100 * dataset_correct / dataset_total
    class_accuracies = 100 * correct_class_counts / np.maximum(total_class_counts, 1)  # Avoid division by zero
    return dataset_accuracy, class_accuracies


def main(dataset_name):
    # Configuration
    config = get_config(dataset_name)
    num_classes = config['num_classes']
    models_dir = os.path.join(config['save_dir'], 'none', dataset_name)
    test_loader = load_test_loader(config, dataset_name)

    # Load model paths
    model_paths = sorted([
        os.path.join(models_dir, fname) for fname in os.listdir(models_dir)
        if fname.endswith('.pth') and '_epoch_200' in fname
    ])
    dataset_accuracies, models = [], []
    class_accuracies = [[] for _ in range(num_classes)]
    print(f'Loaded {len(model_paths)} models.')
    save_dir = os.path.join('Figures/', 'none', dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f'ensemble_accuracies.pdf')

    # Sequentially load models, compute ensemble accuracy as we increase the ensemble size
    for i, model_path in enumerate(model_paths):
        model = load_model(model_path, num_classes)
        models.append(model)
        dataset_acc, class_accs = ensemble_accuracies(models, test_loader, num_classes)

        dataset_accuracies.append(dataset_acc)
        for c in range(num_classes):
            class_accuracies[c].append(class_accs[c])

        print(f"Ensemble size: {i + 1}, Dataset accuracy: {dataset_acc:.2f}%")

    # Plot dataset-level and class-level accuracies as a function of ensemble size
    plt.figure(figsize=(12, 8))
    x_values = range(1, len(models) + 1)
    plt.plot(x_values, dataset_accuracies, label='Dataset-level Accuracy', linewidth=2)

    # Plot each class-level accuracy
    for c in range(num_classes):
        plt.plot(x_values, class_accuracies[c], label=f'Class {c} Accuracy', linestyle='--', alpha=0.7)

    plt.xlabel("Number of Models in Ensemble")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Ensemble Accuracy vs. Number of Models ({dataset_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ensemble accuracy as a function of ensemble size.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    args = parser.parse_args()

    main(args.dataset_name)
