import argparse
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_networks import ResNet18LowRes
from utils import get_config


def load_models(pruning_strategy: str, dataset_name: str) -> Dict[int, List[dict]]:
    """
    Load all models for the specified pruning strategy and dataset.

    :param pruning_strategy: Abbreviated pruning strategy (e.g., 'loop' for leave_one_out_pruning).
    :param dataset_name: Name of the dataset (e.g., 'CIFAR10').
    :return: Dictionary where keys are pruning rates and values are lists of model state dictionaries.
    """
    models_dir = "Models"
    models_by_rate = {}

    if pruning_strategy != 'none':
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


def compute_pruned_percentage(pruning_strategy: str, dataset_name: str,
                              models_by_rate: Dict[int, List[dict]]) -> Dict[int, List[float]]:
    """
    Load the class_level_sample_counts.pkl file for each pruning rate and compute the percentage of data pruned
    per class for the specified dataset and pruning strategy.

    :param pruning_strategy: The pruning strategy used (e.g., 'loop', 'linearaclp').
    :param dataset_name: The name of the dataset (e.g., 'CIFAR10').
    :param models_by_rate: Dictionary where keys are pruning rates and values are lists of model state dictionaries.
                           The pruning rates are used to locate the relevant class-level sample counts.
    :return: Dictionary where keys are pruning rates and values are lists of pruned percentages for each class.
             Each list contains percentages of data pruned per class at the respective pruning rate.
    """
    pruning_rates = models_by_rate.keys()
    pruned_percentages = {pruning_rate: [] for pruning_rate in pruning_rates}

    # Iterate over each pruning rate in models_by_rate
    for pruning_rate in pruning_rates:
        if pruning_rate != 0:
            pkl_path = os.path.join("Results", pruning_strategy + str(pruning_rate), dataset_name,
                                    "class_level_sample_counts.pkl")
            with open(pkl_path, "rb") as file:
                class_level_sample_counts = pickle.load(file)
            remaining_data_count = class_level_sample_counts[pruning_strategy][pruning_rate]
            for c in range(num_classes):
                pruned_percentage = 100.0 * (num_samples[c] - remaining_data_count[c]) / num_samples[c]
                pruned_percentages[pruning_rate].append(pruned_percentage)
        else:
            pruned_percentages[0] = [0.0 for _ in range(num_classes)]

    return pruned_percentages


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
        model = ResNet18LowRes(num_classes)
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
    class_correct = torch.zeros(num_classes)
    class_counts = torch.zeros(num_classes)

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
            for class_id in range(num_classes):
                class_mask = (labels == class_id)
                class_correct[class_id] += (batch_predictions[class_mask] == class_id).sum().item()
                class_counts[class_id] += class_mask.sum().item()

    # Calculate overall accuracy and per-class accuracies
    dataset_accuracy = 100.0 * dataset_correct / len(test_loader.dataset)
    class_accuracies = [(100.0 * class_correct[i] / class_counts[i]).item() for i in range(num_classes)]

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
        model = ResNet18LowRes(num_classes)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()

        # Initialize counters
        dataset_correct = 0
        class_correct = torch.zeros(num_classes)
        class_counts = torch.zeros(num_classes)

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)

                # Dataset-level accuracy
                dataset_correct += (predicted == labels).sum().item()

                # Class-level accuracy
                for class_id in range(num_classes):
                    class_mask = (labels == class_id)
                    class_correct[class_id] += (predicted[class_mask] == class_id).sum().item()
                    class_counts[class_id] += class_mask.sum().item()

        # Compute accuracy for this model
        dataset_accuracy = 100.0 * dataset_correct / len(test_loader.dataset)
        per_class_accuracies = [100.0 * class_correct[i] / class_counts[i] for i in range(num_classes)]

        # Append to accuracy lists
        dataset_accuracies.append(dataset_accuracy)
        class_accuracies.append(per_class_accuracies)

    # Average dataset accuracy and per-class accuracies across all models
    avg_dataset_accuracy = sum(dataset_accuracies) / len(dataset_accuracies)
    avg_class_accuracies = [(sum(acc[i] for acc in class_accuracies) / len(class_accuracies)).item()
                            for i in range(num_classes)]

    return [avg_dataset_accuracy] + avg_class_accuracies


def evaluate_all_ensembles(models_by_rate: Dict[int, List[dict]],
                           test_loader) -> Dict[int, List[List[float]]]:
    """
    Evaluate all ensembles incrementally for stability analysis.

    :param models_by_rate: Dictionary where keys are pruning rates and values are lists of model state dictionaries.
    :param test_loader: DataLoader for the CIFAR-10 test set.
    :return: Dictionary with pruning rates as keys and a list of accuracy dictionaries, each representing the accuracy for incremental ensemble sizes.
    """
    incremental_ensemble_accuracies = {}

    for pruning_rate, ensemble in tqdm(models_by_rate.items(), desc='Iterating over pruning rates.'):
        print(f"Evaluating ensemble for pruning rate {pruning_rate}% with incremental model sizes...")

        # To store results for each incremental ensemble size
        incremental_accuracies = []

        # Evaluate accuracy for ensemble sizes from 1 to the full size of the ensemble (20 in your case)
        for num_models in range(1, len(ensemble) + 1):
            accuracy_for_current_size = evaluate_ensemble(ensemble[:num_models], test_loader)
            incremental_accuracies.append(accuracy_for_current_size)

        # Store the incremental results for this pruning rate
        incremental_ensemble_accuracies[pruning_rate] = incremental_accuracies

    return incremental_ensemble_accuracies


def evaluate_all_models_individually(models_by_rate: Dict[int, List[dict]],
                                     test_loader) -> Dict[int, List[List[float]]]:
    """
    Evaluate individual models incrementally across pruning rates.

    :param models_by_rate: Dictionary where keys are pruning rates and values are lists of model state dictionaries.
    :param test_loader: DataLoader for the CIFAR-10 test set.
    :return: Dictionary with pruning rates as keys and a list of accuracy dictionaries, each representing the accuracy for incremental model counts.
    """
    incremental_individual_accuracies = {}

    for pruning_rate, ensemble in tqdm(models_by_rate.items(), desc='Iterating over pruning rates.'):
        print(f"Evaluating individual models for pruning rate {pruning_rate}% with incremental model sizes...")

        # To store results for each incremental ensemble size
        incremental_accuracies = []

        # Evaluate accuracy for model counts from 1 to the full size of the ensemble
        for num_models in range(1, len(ensemble) + 1):
            accuracy_for_current_size = evaluate_individual_models(ensemble[:num_models], test_loader)
            incremental_accuracies.append(accuracy_for_current_size)

        # Store the incremental results for this pruning rate
        incremental_individual_accuracies[pruning_rate] = incremental_accuracies

    return incremental_individual_accuracies


def plot_ensemble_accuracies(incremental_ensemble_accuracies: Dict[int, List[List[float]]],
                             pruned_percentages: Dict[int, List[float]]):
    """
    Plot class-level and dataset-level accuracies across pruning rates for the full ensemble size.

    :param incremental_ensemble_accuracies: Dictionary with pruning rates as keys and lists of accuracy dictionaries,
                                            where each dictionary contains the accuracies for incremental ensemble sizes.
    :param pruned_percentages: Dictionary where keys are pruning rates and values are lists of pruned percentages for
                               each class. Each list contains percentages of data pruned per class at the respective
                               pruning rate.
    """

    # Sort the pruning rates for consistent plotting
    pruning_parameters = sorted(incremental_ensemble_accuracies.keys())

    # Extract the dataset and class accuracies for the full ensemble (last element in each list)
    dataset_accuracies = [incremental_ensemble_accuracies[pp][-1][0] for pp in pruning_parameters]
    class_accuracies = [
        [incremental_ensemble_accuracies[pp][-1][i + 1] for pp in pruning_parameters]
        for i in range(num_classes)
    ]

    # Create a figure with 10 subplots for each class
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey=True)
    fig.suptitle('Ensemble Accuracies Across Pruning Rates', fontsize=16)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    pruning_rates = [pruned_percentages[pruning_parameter] for pruning_parameter in pruning_parameters]

    # Plot class-level and dataset-level accuracy for each class
    for class_id in range(10):
        class_pruning_rates = [pruning_rates[i][class_id] for i in range(num_classes)]
        if set(class_pruning_rates) == {0.0}:
            class_pruning_rates = pruning_parameters
        ax = axes[class_id]
        ax.plot(class_pruning_rates, class_accuracies[class_id], marker='o')
        ax.plot(class_pruning_rates, dataset_accuracies, linestyle='--', marker='x')

        # Customize each subplot
        ax.set_title(f'Class {class_id} Accuracy')
        ax.set_xlabel('Pruning Rate (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    plt.savefig(os.path.join(figure_save_dir, 'Figure_Full_Ensemble.pdf'))
    plt.close()


def plot_individual_model_accuracies(incremental_individual_accuracies: Dict[int, List[List[float]]],
                                     pruned_percentages: Dict[int, List[float]]):
    """
    Plot class-level and dataset-level accuracies across pruning rates for individual models with the full ensemble size.

    :param incremental_individual_accuracies: Dictionary with pruning rates as keys and lists of accuracy dictionaries,
                                              where each dictionary contains the accuracies for incremental model counts.
    :param pruned_percentages: Dictionary where keys are pruning rates and values are lists of pruned percentages for
                               each class. Each list contains percentages of data pruned per class at the respective
                               pruning rate.
    """
    pruning_parameters = sorted(incremental_individual_accuracies.keys())

    # Extract dataset and class accuracies for the full model count (last element in each list)
    dataset_accuracies = [incremental_individual_accuracies[pp][-1][0] for pp in pruning_parameters]
    class_accuracies = [
        [incremental_individual_accuracies[pp][-1][i + 1] for pp in pruning_parameters]
        for i in range(num_classes)
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey=True)
    fig.suptitle('Individual Model Accuracies Across Pruning Rates', fontsize=16)

    axes = axes.flatten()

    pruning_rates = [pruned_percentages[pruning_parameter] for pruning_parameter in pruning_parameters]

    print(incremental_individual_accuracies)
    print(pruned_percentages)
    print(len(incremental_individual_accuracies[60]), len(incremental_individual_accuracies[60][0]))

    for class_id in range(10):
        class_pruning_rates = [pruning_rates[i][class_id] for i in range(num_classes)]
        if set(class_pruning_rates) == {0.0}:
            class_pruning_rates = pruning_parameters
        ax = axes[class_id]
        print('-'*20)
        print(class_accuracies)
        print(dataset_accuracies)
        print(class_pruning_rates)
        ax.plot(class_pruning_rates, class_accuracies[class_id], marker='o')
        ax.plot(class_pruning_rates, dataset_accuracies, linestyle='--', marker='x')

        ax.set_title(f'Class {class_id} Accuracy')
        ax.set_xlabel('Pruning Rate (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(figure_save_dir, 'Figure_Full_Individual_Models.pdf'))
    plt.close()


def measure_total_ensemble_variability(incremental_results: Dict[int, List[List[float]]]) -> List[List[List[float]]]:
    """
    Measure the total variability of each accuracy metric across incremental ensemble sizes.

    :param incremental_results: Dictionary with pruning rates as keys and lists of accuracy lists.
                                Each list contains accuracy results for incremental ensemble sizes.
    :return: List of lists, where each inner list corresponds to the total variability for each accuracy metric
             at each ensemble size across all pruning rates.
    """
    # Determine the maximum number of ensemble sizes from the incremental results
    max_ensemble_size = max(len(v) for v in incremental_results.values())
    num_metrics = len(next(iter(incremental_results.values()))[0])  # Get number of accuracy metrics

    # Initialize total variability as a list of lists with zeros
    total_variability = [[[0.0 for _ in range(num_metrics)]
                          for _ in range(max_ensemble_size - 1)]
                         for _ in range(len(incremental_results.keys()))]

    for pruning_rate_index, (pruning_rate, results) in enumerate(incremental_results.items()):
        print(f"Calculating variability for pruning rate {pruning_rate}%...")

        # Iterate over ensemble sizes, comparing accuracies for sizes i and i+1
        for i in range(len(results) - 1):
            current_results = results[i]
            next_results = results[i + 1]

            # Calculate variability for each accuracy metric
            accuracy_changes = [abs(next_results[j] - current_results[j]) for j in range(num_metrics)]

            # Add this pruning rate's variability to the total variability for this ensemble size
            for j in range(num_metrics):
                total_variability[pruning_rate_index][i][j] += accuracy_changes[j]

    return total_variability


def plot_total_variability(total_variability: List[List[List[float]]], pruning_rates: List[int], file_name: str):
    """
    Plot the total variability across ensemble sizes for each class and the dataset-level,
    separately for each pruning rate.

    :param total_variability: List of lists of lists where each innermost list contains the variability for each
                              accuracy metric at each ensemble size for each pruning rate.
    :param pruning_rates: List of pruning rates for labeling the plots.
    :param file_name: Name of the file to save the figure.
    """
    ensemble_sizes = range(1, len(total_variability) + 1)  # Ensemble sizes start from 1 to the max size

    # Create a figure with 10 subplots for each class
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey=True)
    fig.suptitle('Total Variability Across Ensemble Sizes by Pruning Rate', fontsize=16)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Normalize pruning rate to grayscale values between 0 (black) and 1 (white)
    max_pruning_rate = max(pruning_rates)
    colors = [(pr / max_pruning_rate) * 0.8 + 0.2 for pr in pruning_rates]  # 0.2 to 1 in grayscale

    # For each class, plot class-level variability per pruning rate
    for class_id in range(num_classes):
        ax = axes[class_id]

        # For each pruning rate, plot the class-level variability
        for pruning_rate_index, (pruning_rate, color) in enumerate(zip(pruning_rates, colors)):
            class_variability = [total_variability[ensemble_size][class_id + 1][pruning_rate_index]
                                 for ensemble_size in range(len(total_variability))]
            # dataset_variability = [total_variability[ensemble_size][0][pruning_rate_index]
            #                        for ensemble_size in range(len(total_variability))]

            # Plot class-level variability for this pruning rate
            ax.plot(ensemble_sizes, class_variability, color=str(color))

        # Customize each subplot
        ax.set_title(f'Class {class_id} Variability')
        ax.set_xlabel('Ensemble Size')
        ax.set_ylabel('Variability')
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    plt.savefig(os.path.join(figure_save_dir, file_name))
    plt.close()


def save_file(save_dir, filename, data):
    os.makedirs(save_dir, exist_ok=True)
    save_location = os.path.join(save_dir, filename)
    with open(save_location, "wb") as file:
        pickle.dump(data, file)


def main(pruning_strategy, dataset_name):
    result_dir = os.path.join("Results", dataset_name, pruning_strategy)
    models = load_models(pruning_strategy, dataset_name)
    test_loader = load_cifar10_test_set(1024)
    pruned_percentages = compute_pruned_percentage(pruning_strategy, dataset_name, models)

    # Evaluate ensemble performance
    if os.path.exists(os.path.join(result_dir, "incremental_ensemble_results.pkl")):
        with open(os.path.join(result_dir, "incremental_ensemble_results.pkl"), 'rb') as f:
            incremental_ensemble_results = pickle.load(f)
    else:
        incremental_ensemble_results = evaluate_all_ensembles(models, test_loader)
        save_file(result_dir, "incremental_ensemble_results.pkl", incremental_ensemble_results)
    plot_ensemble_accuracies(incremental_ensemble_results, pruned_percentages)
    variabilities = measure_total_ensemble_variability(incremental_ensemble_results)
    plot_total_variability(variabilities, list(incremental_ensemble_results.keys()), 'Ensemble_Variability.pdf')

    # Evaluate individual models' performance
    if os.path.exists(os.path.join(result_dir, "incremental_individual_results.pkl")):
        with open(os.path.join(result_dir, "incremental_individual_results.pkl"), 'rb') as f:
            incremental_individual_results = pickle.load(f)
    else:
        incremental_individual_results = evaluate_all_models_individually(models, test_loader)
        save_file(result_dir, "incremental_individual_results.pkl", incremental_individual_results)
    plot_individual_model_accuracies(incremental_individual_results, pruned_percentages)
    variabilities = measure_total_ensemble_variability(incremental_individual_results)
    plot_total_variability(variabilities, list(incremental_individual_results.keys()), 'Individual_Variability.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--pruning_strategy", type=str, required=True,
                        help="Abbreviated pruning strategy (e.g., 'loop' for leave_one_out_pruning)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (e.g., 'CIFAR10')")
    args = parser.parse_args()

    figure_save_dir = os.path.join('Figures/', args.pruning_strategy, args.dataset_name)
    config = get_config(args.dataset_name)
    num_classes = config['num_classes']
    num_samples = [config['num_training_samples'] / num_classes for _ in range(num_classes)]
    os.makedirs(figure_save_dir, exist_ok=True)
    main(args.pruning_strategy, args.dataset_name)
