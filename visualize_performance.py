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


def load_cifar10_test_set(class_grouped_hardness, test_labels, batch_size: int = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
    number_of_blocks = 5

    blocks = {i: [] for i in range(number_of_blocks)}
    for class_index in range(len(class_grouped_hardness)):
        class_indices_within_dataset = np.where(np.array(test_labels) == class_index)[0]
        sorted_class_indices = np.argsort(class_grouped_hardness[class_index])
        sorted_indices = class_indices_within_dataset[sorted_class_indices]
        num_samples = len(sorted_indices)
        block_size = num_samples // number_of_blocks
        class_blocks = [sorted_indices[i * block_size: (i + 1) * block_size] for i in range(number_of_blocks)]
        # This is necessary to handle any remaining samples and put them in the last block. That is because we
        # // number_of_blocks and not / number_of_blocks.
        class_blocks[-1] = np.append(class_blocks[-1], sorted_indices[number_of_blocks * block_size:])
        for block_index, block in enumerate(class_blocks):
            blocks[block_index].extend(list(block))
    print(blocks)
    print()
    for block_index in range(number_of_blocks):
        print(blocks[block_index])

    block_loaders = [DataLoader(Subset(test_set, blocks[block_index]), batch_size=batch_size, shuffle=False)
                     for block_index in range(number_of_blocks)]

    return block_loaders


def evaluate_block(ensemble: List[dict], test_loader, class_index: int, block_index: int, pruning_rate: int, results):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize confusion matrix components
    total_Tp, total_Fp, total_Fn, total_Tn = 0, 0, 0, 0

    for model_state in ensemble:
        model = ResNet18LowRes(num_classes)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)

                for pred, label in zip(predicted, labels):
                    if label.item() == class_index:
                        if pred.item() == class_index:
                            total_Tp += 1
                        else:
                            total_Fn += 1
                    else:
                        if pred.item() == class_index:
                            total_Fp += 1
                        else:
                            total_Tn += 1

    # Average over ensemble
    results['Tp'][pruning_rate][block_index][class_index] = total_Tp / len(ensemble)
    results['Fp'][pruning_rate][block_index][class_index] = total_Fp / len(ensemble)
    results['Fn'][pruning_rate][block_index][class_index] = total_Fn / len(ensemble)
    results['Tn'][pruning_rate][block_index][class_index] = total_Tn / len(ensemble)


def evaluate_ensemble(models_by_rate: Dict[int, List[dict]], test_loaders: List[DataLoader],
                      results: Dict[str, Dict], class_index: int):
    for pruning_rate, ensemble in tqdm(models_by_rate.items(), desc='Iterating over pruning rates.'):
        print(f"Evaluating ensemble for pruning rate {pruning_rate}% with incremental model sizes...")
        if class_index == 0:
            for metric in ['Tp', 'Fp', 'Fn', 'Tn']:
                results[metric][pruning_rate] = [{class_id: 0 for class_id in range(num_classes)}
                                                 for _ in range(len(test_loaders))]
        for block_index in range(len(test_loaders)):
            evaluate_block(ensemble, test_loaders[block_index], class_index, block_index, pruning_rate, results)


def plot_class_level_results(results, pruned_percentages):
    pruning_parameters = sorted(results['Tp'].keys())

    for metric in results.keys():
        # Create a figure with 10 subplots for each class
        fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey=True)
        fig.suptitle(f"Class-Level  {metric} Across Pruning Rates", fontsize=16)

        # Flatten axes for easy iteration
        axes = axes.flatten()

        # Extract pruning rates for each class
        pruning_rates = [pruned_percentages[pruning_parameter] for pruning_parameter in pruning_parameters]
        number_of_blocks = len(results['Tp'][pruning_parameters[0]])

        all_lines = {class_id: [] for class_id in range(num_classes)}
        # Plot class-level metric values for each class
        for block_index in range(number_of_blocks):
            start = round(block_index / number_of_blocks, 2)
            end = round((block_index + 1) / number_of_blocks, 2)
            color = (block_index / (number_of_blocks - 1), 1 - block_index / (number_of_blocks - 1), 0)

            for class_id in range(num_classes):
                class_pruning_rates = [pruning_rates[i][class_id] for i in range(len(pruning_parameters))]
                if set(class_pruning_rates) == {0.0}:
                    class_pruning_rates = pruning_parameters

                y = [results[metric][pruning_parameter][block_index][class_id]
                     for pruning_parameter in pruning_parameters]
                all_lines[class_id].append(y)

                axes[class_id].plot(class_pruning_rates, y, marker='o', color=color, label=f'{start}-{end} Hardness')

        for class_id in range(num_classes):
            class_pruning_rates = [pruning_rates[i][class_id] for i in range(len(pruning_parameters))]
            avg_y = np.mean(all_lines[class_id], axis=0)
            axes[class_id].plot(class_pruning_rates, avg_y, color='blue', label=f'Average Hardness')
            axes[class_id].plot(class_pruning_rates, [avg_y[0] - 0.001 for _ in range(len(avg_y))], color='blue',
                                alpha=0.4)
            axes[class_id].plot(class_pruning_rates, [avg_y[0] + 0.001 for _ in range(len(avg_y))], color='blue',
                                alpha=0.4)
            axes[class_id].set_title(f'Class {class_id} {metric}')
            axes[class_id].set_xlabel('Pruning Rate (%)')
            axes[class_id].set_ylabel(metric)
            axes[class_id].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
        plt.savefig(os.path.join(figure_save_dir, f'Class_Level_{metric}.pdf'))
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
    test_loader = load_cifar10_test_set(class_el2n_scores, test_labels, 1024)

    # Evaluate ensemble performance
    if os.path.exists(os.path.join(result_dir, "ensemble_results.pkl")):
        with open(os.path.join(result_dir, "ensemble_results.pkl"), 'rb') as f:
            results = pickle.load(f)
    else:
        results = {
            'Tp': {},
            'Fn': {},
            'Fp': {},
            'Tn': {}
        }

        for class_index in range(num_classes):
            evaluate_ensemble(models, test_loader, results, class_index)
        save_file(result_dir, "ensemble_results.pkl", results)

    results['F1'], results['MCC'], results['Tn'], results['Average Model Accuracy'] = {}, {}, {}, {}
    results['Precision'], results['Recall'] = {}, {}
    print(pruned_percentages)
    for pruning_rate in results['Tp'].keys():
        results['F1'][pruning_rate], results['MCC'][pruning_rate], results['Tn'][pruning_rate] = [], [], []
        results['Precision'][pruning_rate], results['Recall'][pruning_rate] = [], []
        results['Average Model Accuracy'][pruning_rate] = []
        number_of_blocks = len(results['Tp'][pruning_rate])
        for block_index in range(number_of_blocks):
            block_F1_results, block_MCC_results, block_Tns, block_accuracies = {}, {}, {}, {}
            block_precisions, block_recalls = {}, {}
            for class_id in range(num_classes):
                Tp = results['Tp'][pruning_rate][block_index][class_id]
                Fp = results['Fp'][pruning_rate][block_index][class_id]
                Fn = results['Fn'][pruning_rate][block_index][class_id]
                Tn = sum(num_test_samples) / number_of_blocks - (Tp + Fp + Fn)

                precision = Tp / (Tp + Fp) if (Tp + Fp) > 0 else 0.0
                recall = Tp / (Tp + Fn) if (Tp + Fn) > 0 else 0.0
                accuracy = number_of_blocks * (Tp + Tn) / sum(num_test_samples)
                F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                MCC_numerator = Tp * Tn - Fp * Fn
                MCC_denominator = ((Tp + Fp) * (Tp + Fn) * (Tn + Fp) * (Tn + Fn)) ** 0.5
                MCC = 0.0 if MCC_denominator == 0 else MCC_numerator / MCC_denominator

                block_F1_results[class_id] = F1
                block_MCC_results[class_id] = MCC
                block_Tns[class_id] = Tn
                block_accuracies[class_id] = accuracy
                block_precisions[class_id] = precision
                block_recalls[class_id] = recall

            results['F1'][pruning_rate].append(block_F1_results)
            results['MCC'][pruning_rate].append(block_MCC_results)
            results['Tn'][pruning_rate].append(block_Tns)
            results['Average Model Accuracy'][pruning_rate].append(block_accuracies)
            results['Precision'][pruning_rate].append(block_precisions)
            results['Recall'][pruning_rate].append(block_recalls)

    plot_class_level_results(results, pruned_percentages)


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
