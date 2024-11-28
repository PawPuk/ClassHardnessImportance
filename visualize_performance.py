import argparse
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from neural_networks import ResNet18LowRes
from utils import get_config


def load_models(pruning_strategy: str, dataset_name: str, hardness_type: str) -> Dict[int, List[dict]]:
    """
    Load all models for the specified pruning strategy and dataset.

    :param pruning_strategy: Abbreviated pruning strategy (e.g., 'loop' for leave_one_out_pruning).
    :param dataset_name: Name of the dataset (e.g., 'CIFAR10').
    :param hardness_type: Type of hardness (objective vs subjective)
    :return: Dictionary where keys are pruning rates and values are lists of model state dictionaries.
    """
    models_dir = "Models"
    models_by_rate = {}

    if pruning_strategy != 'none':
        # Walk through each folder in the Models directory
        for root, dirs, files in os.walk(models_dir):
            # Check if the path matches the specified pruning strategy and dataset name
            if f"{pruning_strategy}" in root and f"{dataset_name}" in root and hardness_type in root \
                    and ['unprotected', 'protected'][protect_prototypes]:
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
                                    f"{['unprotected', 'protected'][protect_prototypes]}_class_level_sample_counts.pkl")
            with open(pkl_path, "rb") as file:
                class_level_sample_counts = pickle.load(file)
            remaining_data_count = class_level_sample_counts[pruning_strategy][pruning_rate]
            for c in range(num_classes):
                pruned_percentage = 100.0 * (num_training_samples[c] - remaining_data_count[c]) / num_training_samples[c]
                pruned_percentages[pruning_rate].append(pruned_percentage)
        else:
            pruned_percentages[0] = [0.0 for _ in range(num_classes)]

    return pruned_percentages


def load_el2n_scores(dataset_name):
    with open(os.path.join('Results/', dataset_name, 'el2n_scores.pkl'), 'rb') as file:
        _, _, _, _, all_el2n_scores, class_el2n_scores, test_labels, _ = pickle.load(file)
    hardness = np.mean(np.array(all_el2n_scores), axis=1)
    hardness_grouped_by_class = {class_id: np.mean(np.array(class_scores), axis=1)
                                 for class_id, class_scores in class_el2n_scores.items()}
    return hardness, hardness_grouped_by_class, test_labels


def load_cifar10_test_set(hardness, class_grouped_hardness, test_labels, batch_size: int = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)

    number_of_blocks = 5

    sorted_indices = np.argsort(hardness)
    num_samples = len(sorted_indices)
    block_size = num_samples // number_of_blocks
    blocks = [sorted_indices[i * block_size: (i + 1) * block_size] for i in range(number_of_blocks)]
    # This is necessary to handle any remaining samples and put them in the last block. That is because we
    # // number_of_blocks and not / number_of_blocks.
    blocks[-1] = np.append(blocks[-1], sorted_indices[number_of_blocks * block_size:])
    block_loaders = [DataLoader(Subset(test_set, block), batch_size=batch_size, shuffle=False) for block in blocks]

    class_block_loaders = {}
    for class_index in range(len(class_grouped_hardness)):
        class_indices_within_dataset = np.where(np.array(test_labels) == class_index)[0]
        sorted_class_indices = np.argsort(class_grouped_hardness[class_index])
        sorted_indices = class_indices_within_dataset[sorted_class_indices]
        num_samples = len(sorted_indices)
        block_size = num_samples // number_of_blocks
        blocks = [sorted_indices[i * block_size: (i + 1) * block_size] for i in range(number_of_blocks)]
        blocks[-1] = np.append(blocks[-1], sorted_indices[number_of_blocks * block_size:])
        class_block_loaders[class_index] = [DataLoader(Subset(test_set, block), batch_size=batch_size, shuffle=False)
                                            for block in blocks]

    return block_loaders, class_block_loaders


def evaluate_block(ensemble: List[dict], test_loader) -> Tuple[float, float, int, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_logits, accuracies_of_every_model = [], []

    for model_state in ensemble:
        model = ResNet18LowRes(num_classes)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()

        model_logits, correct = [], 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                model_logits.append(outputs.cpu())

        model_accuracy = 100.0 * correct / len(test_loader.dataset)
        accuracies_of_every_model.append(model_accuracy)
        all_logits.append(torch.cat(model_logits, dim=0))

    avg_logits = torch.stack(all_logits).mean(dim=0)
    correct = 0
    # Here we use logit averaging to measure the accuracy of an ensemble
    _, predicted = avg_logits.max(1)
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            batch_start = i * test_loader.batch_size
            batch_end = batch_start + labels.size(0)
            batch_predictions = predicted[batch_start:batch_end]
            correct += (batch_predictions == labels).sum().item()

    average_model_accuracy = sum(accuracies_of_every_model) / len(accuracies_of_every_model)
    ensemble_accuracy = 100.0 * correct / len(test_loader.dataset)
    return ensemble_accuracy, average_model_accuracy, correct, len(test_loader.dataset) - correct


def evaluate_ensemble(models_by_rate: Dict[int, List[dict]], test_loaders: List[DataLoader],
                      results: Dict[int, Dict[str, Dict]], class_index: int):
    for pruning_rate, ensemble in tqdm(models_by_rate.items(), desc='Iterating over pruning rates.'):
        print(f"Evaluating ensemble for pruning rate {pruning_rate}% with incremental model sizes...")
        ensemble_accuracies, average_model_accuracies = [], []
        true_positives, false_negatives, false_positives = [], [], []
        for block_index in range(len(test_loaders)):
            block_ensemble_accuracy, block_average_model_accuracy, Tp, Fn = evaluate_block(
                ensemble, test_loaders[block_index])
            ensemble_accuracies.append(block_ensemble_accuracy)
            average_model_accuracies.append(block_average_model_accuracy)
            true_positives.append(Tp)
            false_negatives.append(Fn)
        results[class_index]['class_ensemble_results'][pruning_rate] = ensemble_accuracies
        results[class_index]['class_average_model_results'][pruning_rate] = average_model_accuracies
        results[class_index]['Tp'][pruning_rate] = true_positives
        results[class_index]['Fn'][pruning_rate] = false_negatives


def rearrange_results(results):
    new_results = {'class_ensemble_results': {}, 'class_average_model_results': {}, 'Tp': {}, 'Fp': {}}
    for class_id, class_data in results.items():
        for measure in ['class_ensemble_results', 'class_average_model_results', 'Tp', 'Fp']:
            new_results[measure][class_id] = class_data[measure]
    return new_results


def plot_class_level_accuracies(pruning_parameters, class_accuracies, pruned_percentages, mode=0):
    # Create a figure with 10 subplots for each class
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey=True)
    fig.suptitle(f"Class-Level  {['Ensemble Accuracies', 'Average Model Accuracies', 'F1 Scores', 'MCCs']} Across "
                 f"Pruning Rates", fontsize=16)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Extract pruning rates for each class
    pruning_rates = [pruned_percentages[pruning_parameter] for pruning_parameter in pruning_parameters]
    n = len(class_accuracies[0][pruning_parameters[0]])

    # Plot class-level accuracy for each class
    for class_id in range(10):
        class_pruning_rates = [pruning_rates[i][class_id] for i in range(len(pruning_parameters))]
        if set(class_pruning_rates) == {0.0}:
            class_pruning_rates = pruning_parameters
        ax = axes[class_id]
        all_lines = []
        for block_index in range(n):
            start = round(block_index / n, 2)
            end = round((block_index + 1) / n, 2)
            y = [class_accuracies[class_id][pruning_parameter][block_index] for pruning_parameter in pruning_parameters]
            all_lines.append(y)
            color = (block_index / (n - 1), 1 - block_index / (n - 1), 0)
            ax.plot(class_pruning_rates, y, marker='o', color=color, label=f'{start}-{end} Hardness')

        average_line = [sum(values) / len(values) for values in zip(*all_lines)]
        ax.plot(class_pruning_rates, average_line, color='blue', marker='o', linestyle='--', linewidth=2,
                label='Average')

        # Customize each subplot
        ax.set_title(f'Class {class_id} Accuracy')
        ax.set_xlabel('Pruning Rate (%)')
        s = 'Accuracy (%)' if mode in [0, 1] else 'F1 Score' if mode == 2 else 'MCC'
        ax.set_ylabel(s)
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    if mode == 0:
        s = 'Class_Level_Ensemble_Accuracies.pdf'
    elif mode == 1:
        s = 'Class_Level_Average_Model_Accuracies.pdf'
    elif mode == 2:
        s = 'Class_Level_F1_Scores.pdf'
    else:
        s = 'Class_Level_MCC_Scores.pdf'
    plt.savefig(os.path.join(figure_save_dir, s))
    plt.close()


def save_file(save_dir, filename, data):
    os.makedirs(save_dir, exist_ok=True)
    save_location = os.path.join(save_dir, filename)
    with open(save_location, "wb") as file:
        pickle.dump(data, file)


def main(pruning_strategy, dataset_name, hardness_type):
    result_dir = os.path.join("Results", dataset_name, f"{['unprotected', 'protected'][protect_prototypes]}_"
                                                       f"{hardness_type}_{pruning_strategy}")
    models = load_models(pruning_strategy, dataset_name, hardness_type)
    pruned_percentages = compute_pruned_percentage(pruning_strategy, dataset_name, models)
    all_el2n_scores, class_el2n_scores, test_labels = load_el2n_scores(dataset_name)
    test_loader, class_test_loaders = load_cifar10_test_set(all_el2n_scores, class_el2n_scores, test_labels, 1024)

    # Evaluate ensemble performance
    if os.path.exists(os.path.join(result_dir, "ensemble_results.pkl")):
        with open(os.path.join(result_dir, "ensemble_results.pkl"), 'rb') as f:
            results = pickle.load(f)
    else:
        results = {class_id: {
            'class_ensemble_results': {},
            'class_average_model_results': {},
            'Tp': {},
            'Fn': {},
            'Fp': {}
        } for class_id in range(num_classes)}

        for class_index in range(num_classes):
            evaluate_ensemble(models, class_test_loaders[class_index], results, class_index)
        save_file(result_dir, "ensemble_results.pkl", results)

    results = rearrange_results(results)

    results['F1'], results['MCC'] = {}, {}
    for class_id in range(num_classes):
        results['F1'][class_id], results['MCC'][class_id] = {}, {}
        for pruning_rate in results['Tp'][class_id]:
            results['F1'][class_id][pruning_rate], results['MCC'][class_id][pruning_rate] = [], []
            for block_index in range(len(results['Tp'][class_id][pruning_rate])):
                Tp = results['Tp'][class_id][pruning_rate][block_index]
                Fp = results['Fp'][class_id][pruning_rate][block_index]
                total_positives_c = num_test_samples[class_id] // len(results['Tp'][class_id][pruning_rate])

                Fn = total_positives_c - Tp
                Tn = sum(num_test_samples) // len(results['Tp'][class_id][pruning_rate]) - (Tp + Fp + Fn)
                precision = Tp / (Tp + Fp) if (Tp + Fp) > 0 else 0.0
                recall = Tp / (Tp + Fn) if (Tp + Fn) > 0 else 0.0

                F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                MCC_numerator = Tp * Tn - Fp * Fn
                MCC_denominator = ((Tp + Fp) * (Tp + Fn) * (Tn + Fp) * (Tn + Fn)) ** 0.5
                MCC = 0.0 if MCC_denominator == 0 else MCC_numerator / MCC_denominator

                results['F1'][class_id][pruning_rate].append(F1)
                results['MCC'][class_id][pruning_rate].append(MCC)

    # We sort the pruning rates for consistent plotting
    pruning_parameters = sorted(results['class_ensemble_results'][0].keys())
    plot_class_level_accuracies(pruning_parameters, results['class_ensemble_results'], pruned_percentages)
    plot_class_level_accuracies(pruning_parameters, results['class_average_model_results'], pruned_percentages, 1)
    plot_class_level_accuracies(pruning_parameters, results['F1'], pruned_percentages, 2)
    plot_class_level_accuracies(pruning_parameters, results['MCC'], pruned_percentages, 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--pruning_strategy", type=str, required=True,
                        help="Abbreviated pruning strategy (e.g., 'loop' for leave_one_out_pruning)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (e.g., 'CIFAR10')")
    parser.add_argument('--protect_prototypes', action='store_true',
                        help="Raise this flag to protect the prototypes from pruning - don't prune 1% of the easiest "
                             "samples.")
    parser.add_argument('--hardness_type', type=str, choices=['objective', 'subjective'],
                        help="If set to 'subjective', each model will use the hardness of probe network obtained using "
                             "the same seed (similar to self-paced learning). For 'objective', the average hardness "
                             "computed using all probe networks is used (similar to transfer learning).")

    args = parser.parse_args()

    protect_prototypes = args.protect_prototypes
    figure_save_dir = os.path.join('Figures/', f"{['unprotected', 'protected'][protect_prototypes]}_"
                                               f"{args.hardness_type}_{args.pruning_strategy}", args.dataset_name)
    config = get_config(args.dataset_name)
    num_classes = config['num_classes']
    num_training_samples = config['num_training_samples']
    num_test_samples = config['num_test_samples']
    os.makedirs(figure_save_dir, exist_ok=True)
    main(args.pruning_strategy, args.dataset_name, args.hardness_type)
