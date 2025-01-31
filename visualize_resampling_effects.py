import argparse
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from data import load_dataset
from neural_networks import ResNet18LowRes


def load_models(dataset_name: str) -> Dict[Tuple[str, str, str], List[dict]]:
    """
    Load all models for the specified dataset, organizing by oversampling and undersampling strategies,
    and differentiating between cleaned and uncleaned datasets.

    :param dataset_name: Name of the dataset (e.g., 'CIFAR10').
    :return: Dictionary where keys are tuples (oversampling_strategy, undersampling_strategy, dataset_type)
             and values are lists of model state dictionaries.
    """
    models_dir, models_by_strategy = "Models", {}

    for root, dirs, files in os.walk(models_dir):
        if f"unclean{dataset_name}" in root and 'over_' in root and '_under_' in root:
            dataset_type = 'unclean'
        elif f"clean{dataset_name}" in root and 'over_' in root and '_under_' in root:
            dataset_type = 'clean'
        else:
            continue

        try:
            oversampling_strategy = root.split("over_")[1].split("_under_")[0]
            undersampling_strategy = root.split("_under_")[1].split("_size_")[0]
            key = (oversampling_strategy, undersampling_strategy, dataset_type)
            models_by_strategy.setdefault(key, [])

            for file in files:
                if file.endswith(".pth") and "_epoch_200" in file:
                    model_path = os.path.join(root, file)
                    model_state = torch.load(model_path)
                    models_by_strategy[key].append(model_state)
            if len(models_by_strategy[key]) > 0:
                print(f"Loaded {len(models_by_strategy[key])} models for strategies {key}.")
        except (IndexError, ValueError):
            # Skip directories or files that don't match the expected pattern
            continue

    # Also load models trained on the full dataset (no resampling)
    for dataset_type in ['unclean', 'clean']:
        full_dataset_dir = os.path.join(models_dir, "none", f"{dataset_type}{dataset_name}")
        if os.path.exists(full_dataset_dir):
            key = ('none', 'none', dataset_type)
            models_by_strategy[key] = []
            for file in os.listdir(full_dataset_dir):
                if file.endswith(".pth") and "_epoch_200" in file:
                    model_path = os.path.join(full_dataset_dir, file)
                    model_state = torch.load(model_path)
                    models_by_strategy[key].append(model_state)
                    print(f"Loaded model for full dataset ({dataset_type}): {model_path}")

    print([key for key in models_by_strategy.keys()])
    print(f"Loaded {len(models_by_strategy.keys())} models for {dataset_name}.")
    return models_by_strategy


def load_el2n_scores(dataset_name):
    with open(os.path.join('Results/', f'unclean{dataset_name}', 'AUM.pkl'), 'rb') as file:
        AUM_over_epochs_and_models = pickle.load(file)
    for model_idx, model_list in enumerate(AUM_over_epochs_and_models):
        AUM_over_epochs_and_models[model_idx] = [sample for sample in model_list if len(sample) > 0]

    AUM_scores_over_models = [
        [
            sum(model_list[sample_idx][epoch_idx] for epoch_idx in range(num_epochs)) / num_epochs
            for sample_idx in range(len(AUM_over_epochs_and_models[0]))
        ]
        for model_list in AUM_over_epochs_and_models
    ]
    AUM_scores = np.mean(AUM_scores_over_models, axis=0)

    _, training_dataset, _, _ = load_dataset(dataset_name, False, 42, True)

    hardnesses_by_class, hardness_of_classes, labels = {class_id: [] for class_id in range(num_classes)}, {}, []
    for i, (_, label, _) in enumerate(training_dataset):
        hardnesses_by_class[label].append(AUM_scores[i])
    for label in range(num_classes):
        hardness_of_classes[label] = np.mean(hardnesses_by_class[label])
    return hardnesses_by_class


def load_test_set(class_grouped_hardness, dataset_name, batch_size: int = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std'])
    ])
    if dataset_name == 'CIFAR10':
        test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        test_set = CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        raise Exception

    block_loaders = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return block_loaders


def evaluate_block(ensemble: List[dict], test_loader, class_index: int, block_index: int,
                   strategies: Tuple[str, str, str], results):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    results['Tp'][strategies][block_index][class_index] = total_Tp / len(ensemble)
    results['Fp'][strategies][block_index][class_index] = total_Fp / len(ensemble)
    results['Fn'][strategies][block_index][class_index] = total_Fn / len(ensemble)
    results['Tn'][strategies][block_index][class_index] = total_Tn / len(ensemble)


def evaluate_ensemble(models: Dict[Tuple[str, str, str], List[dict]], test_loader: DataLoader,
                      results: Dict[str, Dict], class_index: int):
    for (over, under, cleanliness), ensemble in models.items():
        print(f"Evaluating ensemble for oversampling strategy {over} and undersampling strategy {under} trained on "
              f"{cleanliness} data.")
        if class_index == 0:
            for metric in ['Tp', 'Fp', 'Fn', 'Tn']:
                results[metric][(over, under, cleanliness)] = [{class_id: 0 for class_id in range(num_classes)}]
        evaluate_block(ensemble, test_loader, class_index, 0, (over, under, cleanliness), results)


def save_file(save_dir, filename, data):
    os.makedirs(save_dir, exist_ok=True)
    save_location = os.path.join(save_dir, filename)
    with open(save_location, "wb") as file:
        pickle.dump(data, file)


def get_class_order(results, base_strategy):
    """
    Determine class order based on average accuracies for the specified base strategy.

    :param results: Results dictionary containing accuracy values.
    :param base_strategy: The strategy used to determine the class order (e.g., ('none', 'none')).
    :return: List of class IDs sorted by ascending average accuracy for the base strategy.
    """
    accuracies = results['Recall'][base_strategy]
    avg_accuracies = {class_id: np.mean([block[class_id] for block in accuracies]) for class_id in range(num_classes)}
    # Sort classes by ascending accuracy
    sorted_classes = sorted(avg_accuracies.keys(), key=lambda k: avg_accuracies[k])
    return sorted_classes


def plot_all_accuracies_sorted(results, class_order, save_path=None):
    """
    Plot the average accuracies for all strategies with classes sorted by a specific order.

    :param results: The results dictionary containing accuracy values.
    :param class_order: List specifying the order of classes for plotting.
    :param save_path: Path to save the plot. If None, the plot will be shown.
    """

    # Plot for all strategies using line segments
    plt.figure(figsize=(12, 8))

    color_palette = plt.cm.tab10(np.linspace(0, 1, len(results['Recall'])))

    for idx, (strategy, accuracies) in enumerate(results['Recall'].items()):
        avg_accuracies = {class_id: np.mean([block[class_id] for block in accuracies]) for class_id in
                          range(num_classes)}

        reordered_accuracies = [avg_accuracies[class_id] for class_id in class_order]

        for reordered_idx, avg_accuracy in enumerate(reordered_accuracies):
            x_start = reordered_idx - 0.4 + (idx / len(results['Recall'])) * 0.8
            x_end = reordered_idx + 0.4 - (idx / len(results['Recall'])) * 0.8
            plt.plot([x_start, x_end], [avg_accuracy * 100, avg_accuracy * 100],
                     color=color_palette[idx], lw=2)

        plt.plot([], [], color=color_palette[idx], lw=2, label=f"{strategy[0]}-{strategy[1]}-{strategy[2]}")

    plt.xlabel("Class (Sorted by Base Strategy)", fontsize=12)
    plt.xticks([])
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Average Accuracy by Class for All Strategies (Sorted by Base Strategy)", fontsize=14)
    plt.legend(fontsize=10, title="Strategies", loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

    # Create a new plot for specific strategies using classical function-like visual style
    filtered_keys = [('random', 'easy', 'clean'), ('hard', 'easy', 'clean'), ('easy', 'easy', 'clean'),
                     ('SMOTE', 'easy', 'clean'), ('none', 'none', 'clean'), ('none', 'none', 'unclean')]
    filtered_results = {key: results['Recall'][key] for key in filtered_keys if key in results['Recall']}

    plt.figure(figsize=(12, 8))
    for idx, (strategy, accuracies) in enumerate(filtered_results.items()):
        avg_accuracies = {class_id: np.mean([block[class_id] for block in accuracies]) for class_id in
                          range(len(class_order))}
        if strategy == ('none', 'none', 'unclean'):
            avg_base_accuracies = avg_accuracies
        reordered_accuracies = [avg_accuracies[class_id] for class_id in class_order]
        plt.plot(range(len(class_order)), [acc * 100 for acc in reordered_accuracies],
                 label=f"{strategy[0]}-{strategy[1]}-{strategy[2]}", lw=1)

    plt.xlabel("Class (Sorted by the Accuracy of Base Strategy)", fontsize=12)
    plt.xticks([])
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Average Accuracy by Class for Filtered Strategies", fontsize=14)
    plt.legend(fontsize=10, title="Strategies", loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        filtered_save_path = save_path.replace('.pdf', '_filtered.pdf')
        plt.savefig(filtered_save_path, bbox_inches='tight')
        print(f"Filtered plot saved to {filtered_save_path}")
    else:
        plt.show()

    # Create a new plot for respective performance of each resampling strategy (in respect to no resampling)
    filtered_keys = [('random', 'easy', 'clean'), ('hard', 'easy', 'clean'), ('easy', 'easy', 'clean'),
                     ('SMOTE', 'easy', 'clean'), ('none', 'none', 'clean')]
    filtered_results = {key: results['Recall'][key] for key in filtered_keys if key in results['Recall']}

    plt.figure(figsize=(12, 8))

    for idx, (strategy, accuracies) in enumerate(filtered_results.items()):
        avg_accuracies = {class_id: np.mean([block[class_id] for block in accuracies]) - avg_base_accuracies[class_id]
                          for class_id in range(len(class_order))}

        sorted_items = sorted(avg_accuracies.items(), key=lambda x: x[1])
        sorted_class_order, sorted_accuracies = zip(*sorted_items)

        reordered_accuracies = [avg_accuracies[class_id] for class_id in sorted_class_order]
        mean_avg_accuracy = np.mean(reordered_accuracies)

        plt.plot(range(len(class_order)), [acc * 100 for acc in reordered_accuracies],
                 label=f"{strategy[0]}-{strategy[1]}-{strategy[2]}", lw=1)
        plt.axhline(mean_avg_accuracy * 100, color=plt.gca().lines[-1].get_color(), linestyle='--',
                    label=f"Dataset-level accuracy ({strategy[0]}-{strategy[1]}-{strategy[2]})")

    plt.xlabel("Class (Sorted by Respective Accuracy Boost)", fontsize=12)
    plt.xticks([])
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Average Accuracy by Class for Filtered Strategies", fontsize=14)
    plt.legend(fontsize=10, title="Strategies", loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        filtered_save_path = save_path.replace('.pdf', '_filtered_respective.pdf')
        plt.savefig(filtered_save_path, bbox_inches='tight')
        print(f"Filtered plot saved to {filtered_save_path}")
    else:
        plt.show()


def main(dataset_name):
    result_dir = os.path.join("Results", dataset_name)
    models = load_models(dataset_name)
    class_el2n_scores = load_el2n_scores(dataset_name)
    test_loader = load_test_set(class_el2n_scores, dataset_name, 1024)

    # Evaluate ensemble performance
    if os.path.exists(os.path.join(result_dir, "resampling_results.pkl")):
        with open(os.path.join(result_dir, "resampling_results.pkl"), 'rb') as f:
            results = pickle.load(f)
    else:
        results = {'Tp': {}, 'Fn': {}, 'Fp': {}, 'Tn': {}}

        for class_index in tqdm(range(num_classes), desc='Iterating through classes'):
            evaluate_ensemble(models, test_loader, results, class_index)
        save_file(result_dir, "resampling_results.pkl", results)

    results['F1'], results['MCC'], results['Tn'], results['Average Model Accuracy'] = {}, {}, {}, {}
    results['Precision'], results['Recall'] = {}, {}
    for strategies in results['Tp'].keys():
        results['F1'][strategies], results['MCC'][strategies], results['Tn'][strategies] = [], [], []
        results['Precision'][strategies], results['Recall'][strategies] = [], []
        results['Average Model Accuracy'][strategies] = []
        number_of_blocks = len(results['Tp'][strategies])
        for block_index in range(number_of_blocks):
            block_F1_results, block_MCC_results, block_Tns, block_accuracies = {}, {}, {}, {}
            block_precisions, block_recalls = {}, {}
            for class_id in range(num_classes):
                Tp = results['Tp'][strategies][block_index][class_id]
                Fp = results['Fp'][strategies][block_index][class_id]
                Fn = results['Fn'][strategies][block_index][class_id]
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

            results['F1'][strategies].append(block_F1_results)
            results['MCC'][strategies].append(block_MCC_results)
            results['Tn'][strategies].append(block_Tns)
            results['Average Model Accuracy'][strategies].append(block_accuracies)
            results['Precision'][strategies].append(block_precisions)
            results['Recall'][strategies].append(block_recalls)

    base_strategy = ('none', 'none', 'unclean')
    class_order = get_class_order(results, base_strategy)
    plot_all_accuracies_sorted(results, class_order, os.path.join(figure_save_dir, 'resampling_effects.pdf'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (e.g., 'CIFAR10')")

    args = parser.parse_args()

    figure_save_dir = os.path.join('Figures/', args.dataset_name)
    config = get_config(args.dataset_name)
    num_classes = config['num_classes']
    num_training_samples = config['num_training_samples']
    num_test_samples = config['num_test_samples']
    num_epochs = config['num_epochs']
    os.makedirs(figure_save_dir, exist_ok=True)
    main(args.dataset_name)
