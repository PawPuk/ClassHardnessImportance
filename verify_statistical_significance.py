import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

from neural_networks import ResNet18LowRes
from utils import get_config, load_dataset


# Measure the overlap between the pruned subsets as a function of hardness estimator and threshold (percentage of data
# pruned). Present results in a 2D heatmap with y being the threshold and x being the ensemble size (overlap between
# the subset pruned by ensemble with x_i models and x_{i+1} models). This should give 3 heatmaps.


def create_model():
    model = ResNet18LowRes(num_classes=NUM_CLASSES)
    return model


def compute_el2n(model, dataloader):
    model.eval()
    el2n_scores = []

    # Accuracy is another way to estimate class-level hardness, so we also compute it
    class_correct = {i: 0 for i in range(NUM_CLASSES)}
    class_total = {i: 0 for i in range(NUM_CLASSES)}

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            # This computes the EL2N scores
            softmax_outputs = F.softmax(outputs, dim=1)
            one_hot_labels = F.one_hot(labels, num_classes=NUM_CLASSES).float()
            l2_errors = torch.norm(softmax_outputs - one_hot_labels, dim=1)
            el2n_scores.extend(l2_errors.cpu().numpy())

            # This computes the class-level accuracies
            _, predicted = torch.max(outputs, 1)
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    class_accuracies = {k: class_correct[k] / class_total[k] if class_total[k] > 0 else 0 for k in class_correct}

    return el2n_scores, class_accuracies


def collect_el2n_scores(loader):
    all_el2n_scores, model_class_accuracies = [], []
    for model_id in range(NUM_MODELS):
        model = create_model().cuda()
        model_path = os.path.join(MODEL_DIR, 'none', f"{DATA_CLEANLINESS}{args.dataset_name}",
                                  f'model_{model_id}_epoch_{SAVE_EPOCH}.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            el2n_scores, class_accuracies = compute_el2n(model, loader)
        else:
            # This code can only be run if models were pretrained. If no pretrained models are found, throw error.
            raise Exception(f'Model {model_id} not found at epoch {SAVE_EPOCH}.')
        all_el2n_scores.append(el2n_scores)
        model_class_accuracies.append(class_accuracies)
    return all_el2n_scores, model_class_accuracies


def save_el2n_scores(el2n_scores):
    with open(os.path.join(RESULTS_SAVE_DIR, 'el2n_scores.pkl'), 'wb') as file:
        pickle.dump(el2n_scores, file)


def load_results(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def get_pruned_indices(el2n_scores, aum_scores, forgetting_scores):
    num_models = len(el2n_scores)  # number of models in the ensemble
    results = {}

    for metric_name, metric_scores in tqdm([("el2n", el2n_scores), ("aum", aum_scores),
                                            ("forgetting", forgetting_scores)], desc='Iterating through metrics.'):
        metric_scores = np.array(metric_scores)
        results[metric_name] = []

        for thresh in THRESHOLDS:
            prune_count = int((thresh / 100) * NUM_SAMPLES)
            metric_results = []

            for num_ensemble_models in range(1, num_models + 1):
                avg_hardness_scores = np.mean(metric_scores[:num_ensemble_models], axis=0)
                # For AUM hard samples have lower values. For EL2N and forgetting the opposite is the case.
                if metric_name == 'aum':
                    sorted_indices = np.argsort(-avg_hardness_scores)
                else:
                    sorted_indices = np.argsort(avg_hardness_scores)
                pruned_indices = sorted_indices[:prune_count]
                metric_results.append(pruned_indices.tolist())
            results[metric_name].append(metric_results)
    return results


def compute_and_visualize_stability_of_pruning(results):
    metric_names = list(results.keys())

    for metric in metric_names:
        metric_results = results[metric]
        num_thresholds = len(metric_results)
        num_models = len(metric_results[0]) - 1
        stability_results = np.zeros((num_thresholds, num_models))

        for i in range(num_thresholds):  # Loop over thresholds (rows)
            for j in range(num_models):  # Loop over model pairs (columns)

                set1 = set(metric_results[i][j])  # Pruned indices for ensemble with j models
                set2 = set(metric_results[i][j + 1])  # Pruned indices for ensemble with j+1 models

                # Compute stability
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                overlap = intersection / union if union > 0 else 0.0
                stability_results[i, j] = overlap

        plt.figure(figsize=(10, 6))
        sns.heatmap(stability_results, annot=True, cmap='coolwarm', cbar_kws={'label': 'Jaccard Overlap'})
        plt.title(f'Overlap Heatmap for {metric.upper()}')
        plt.xlabel('Number of Models in Ensemble (j)')
        plt.ylabel('Pruning Threshold (%)')
        plt.xticks(np.arange(num_models) + 0.5, np.arange(1, num_models + 1))
        plt.yticks(np.arange(num_thresholds) + 0.5, np.arange(10, 100, 10))
        plt.savefig(os.path.join(FIGURES_SAVE_DIR, f'pruning_stability_based_on_{metric}.pdf'))


def compute_overlap_heatmap(results):
    metric_names = list(results.keys())
    num_metrics = len(metric_names)

    plt.figure(figsize=(10, 6))
    for i in range(num_metrics):
        for j in range(i + 1, num_metrics):  # Only compute unique pairs.
            overlaps = []
            for t, thresh in enumerate(THRESHOLDS):
                set1 = set(results[metric_names[i]][t][-1])  # Using the full ensemble.
                set2 = set(results[metric_names[j]][t][-1])
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                overlap = intersection / union if union > 0 else 0.0
                overlaps.append(overlap)

            # Plot overlaps as a function of thresholds.
            plt.plot(THRESHOLDS, overlaps, label=f"{metric_names[i]} vs {metric_names[j]}", marker='o')

    # Customizing the plot.
    plt.xlabel("Pruning Threshold (%)")
    plt.ylabel("Overlap Percentage (%)")
    plt.title("Overlap of Pruned Sets Across Hardness Estimators")
    plt.legend(title="Metric Pairs")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_SAVE_DIR, f'overlap_across_hardness_estimators.pdf'))


def compute_effect_of_ensemble_size_on_resampling(el2n_scores, aum_scores, forgetting_scores, dataset):
    num_models = len(el2n_scores)  # number of models in the ensemble
    results = {}
    for metric_name, metric_scores in tqdm([("el2n", el2n_scores), ("aum", aum_scores),
                                            ("forgetting", forgetting_scores)], desc='Iterating through metrics.'):
        metric_scores = np.array(metric_scores)
        results[metric_name] = []
        for num_ensemble_models in range(2, num_models + 1):
            prev_avg_hardness_scores = np.mean(metric_scores[:num_ensemble_models - 1], axis=0)
            curr_avg_hardness_scores = np.mean(metric_scores[:num_ensemble_models], axis=0)
            prev_class_results, curr_class_results = [[] for _ in range(NUM_CLASSES)], [[] for _ in range(NUM_CLASSES)]
            prev_ratios, curr_ratios = {}, {}

            for i, (_, label, _) in enumerate(dataset):
                prev_class_results[label].append(prev_avg_hardness_scores[i])
                curr_class_results[label].append(curr_avg_hardness_scores[i])
            for label in range(NUM_CLASSES):
                prev_ratios[label] = np.mean(prev_class_results[label])
                curr_ratios[label] = np.mean(curr_class_results[label])

            prev_normalized_ratios = {class_id: ratio / sum(prev_ratios.values())
                                      for class_id, ratio in prev_ratios.items()}
            curr_normalized_ratios = {class_id: ratio / sum(curr_ratios.values())
                                      for class_id, ratio in curr_ratios.items()}

            prev_samples_per_class = {class_id: int(round(prev_normalized_ratio * NUM_SAMPLES))
                                      for class_id, prev_normalized_ratio in prev_normalized_ratios.items()}
            curr_samples_per_class = {class_id: int(round(curr_normalized_ratio * NUM_SAMPLES))
                                      for class_id, curr_normalized_ratio in curr_normalized_ratios.items()}

            differences = [abs(curr_samples_per_class[l] - prev_samples_per_class[l]) for l in range(NUM_CLASSES)]
            relative_differences = [differences[l] / prev_samples_per_class[l] for l in range(NUM_CLASSES)]
            results[metric_name].append((num_ensemble_models, differences, relative_differences))
    return results


def visualize_stability_of_resampling(results):

    metrics = list(results.keys())
    ensemble_sizes = {metric: [entry[0] for entry in results[metric]] for metric in metrics}
    differences = {metric: [entry[1] for entry in results[metric]] for metric in metrics}
    relative_differences = {metric: [entry[2] for entry in results[metric]] for metric in metrics}

    def compute_stats(diff_list):
        max_vals = [np.max(diff) for diff in diff_list]
        min_vals = [np.min(diff) for diff in diff_list]
        avg_vals = [np.mean(diff) for diff in diff_list]
        return max_vals, min_vals, avg_vals

    plt.figure(figsize=(15, 8))
    for metric in metrics:
        max_diffs, min_diffs, avg_diffs = compute_stats(differences[metric])
        plt.plot(ensemble_sizes[metric], max_diffs, label=f'{metric.upper()} Max Diff', linestyle='-', marker='o')
        plt.plot(ensemble_sizes[metric], min_diffs, label=f'{metric.upper()} Min Diff', linestyle='--', marker='x')
        plt.plot(ensemble_sizes[metric], avg_diffs, label=f'{metric.upper()} Avg Diff', linestyle=':', marker='s')
    plt.title("Absolute Differences Across Ensemble Sizes for Hardness Estimators")
    plt.xlabel('Ensemble Size')
    plt.ylabel("Absolute Difference")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(FIGURES_SAVE_DIR, f'absolute_differences_across_ensemble_sizes.pdf'))

    plt.figure(figsize=(15, 8))
    for metric in metrics:
        max_rel_diffs, min_rel_diffs, avg_rel_diffs = compute_stats(relative_differences[metric])
        plt.plot(ensemble_sizes[metric], max_rel_diffs, label=f'{metric.upper()} Max Rel Diff', linestyle='-',
                 marker='o')
        plt.plot(ensemble_sizes[metric], min_rel_diffs, label=f'{metric.upper()} Min Rel Diff', linestyle='--',
                 marker='x')
        plt.plot(ensemble_sizes[metric], avg_rel_diffs, label=f'{metric.upper()} Avg Rel Diff', linestyle=':',
                 marker='s')
    plt.title("Relative Differences Across Ensemble Sizes for Hardness Estimators")
    plt.xlabel('Ensemble Size')
    plt.ylabel("Relative Difference")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(FIGURES_SAVE_DIR, f'relative_differences_across_ensemble_sizes.pdf'))

    # I produce also the below Figures to allow for more clear analysis of the results (metrics have different values)
    for metric in metrics:
        max_diffs, min_diffs, avg_diffs = compute_stats(differences[metric])
        plt.figure(figsize=(10, 6))
        plt.plot(ensemble_sizes[metric], max_diffs, label=f'Max Diff', linestyle='-', marker='o')
        plt.plot(ensemble_sizes[metric], min_diffs, label=f'Min Diff', linestyle='--', marker='x')
        plt.plot(ensemble_sizes[metric], avg_diffs, label=f'Avg Diff', linestyle=':', marker='s')
        plt.title(f"Absolute Differences for {metric.upper()}")
        plt.xlabel('Ensemble Size')
        plt.ylabel("Absolute Difference")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(FIGURES_SAVE_DIR, f'absolute_differences_{metric}.pdf'))

        max_rel_diffs, min_rel_diffs, avg_rel_diffs = compute_stats(relative_differences[metric])
        plt.figure(figsize=(10, 6))
        plt.plot(ensemble_sizes[metric], max_rel_diffs, label=f'Max Rel Diff', linestyle='-', marker='o')
        plt.plot(ensemble_sizes[metric], min_rel_diffs, label=f'Min Rel Diff', linestyle='--', marker='x')
        plt.plot(ensemble_sizes[metric], avg_rel_diffs, label=f'Avg Rel Diff', linestyle=':', marker='s')
        plt.title(f"Relative Differences for {metric.upper()}")
        plt.xlabel('Ensemble Size')
        plt.ylabel("Relative Difference")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(FIGURES_SAVE_DIR, f'relative_differences_{metric}.pdf'))


def main():
    training_loader, _, _, _ = load_dataset(args.dataset_name, args.remove_noise, SEED, False)
    training_all_el2n_scores, training_class_accuracies = collect_el2n_scores(training_loader)
    save_el2n_scores((training_all_el2n_scores, training_class_accuracies))

    aum_path = os.path.join(HARDNESS_SAVE_DIR, 'AUM.pkl')
    forgetting_path = os.path.join(HARDNESS_SAVE_DIR, 'Forgetting.pkl')
    AUM_over_epochs_and_models = load_results(aum_path)
    AUM_scores = [
        [
            sum(model_list[sample_idx][epoch_idx] for epoch_idx in range(NUM_EPOCHS)) / NUM_EPOCHS
            for sample_idx in range(NUM_SAMPLES)
        ]
        for model_list in AUM_over_epochs_and_models
    ]

    forgetting_scores = load_results(forgetting_path)
    print(len(AUM_scores), len(AUM_scores[0]))
    print(len(forgetting_scores), len(forgetting_scores[0]))
    print(len(training_all_el2n_scores), len(training_all_el2n_scores[0]))
    print()

    pruned_indices = get_pruned_indices(training_all_el2n_scores, AUM_scores, forgetting_scores)
    compute_and_visualize_stability_of_pruning(pruned_indices)
    compute_overlap_heatmap(pruned_indices)

    differences = compute_effect_of_ensemble_size_on_resampling(
        training_all_el2n_scores, AUM_scores, forgetting_scores, training_loader.dataset)
    visualize_stability_of_resampling(differences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Hardness of Dataset Samples')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='Specify the dataset name (default: CIFAR10)')
    parser.add_argument('--remove_noise', action='store_true', help='Raise this flag to remove noise from the data.')
    args = parser.parse_args()

    NUM_CLASSES = get_config(args.dataset_name)['num_classes']
    # Modify the below to extract this from the folder (check how many models were trained)
    NUM_MODELS = get_config(args.dataset_name)['num_models']
    NUM_EPOCHS = get_config(args.dataset_name)['num_epochs']
    NUM_SAMPLES = sum(get_config(args.dataset_name)['num_training_samples'])
    MODEL_DIR = get_config(args.dataset_name)['save_dir']
    SAVE_EPOCH = get_config(args.dataset_name)['save_epoch']
    DATA_CLEANLINESS = 'clean' if args.remove_noise else 'unclean'

    RESULTS_SAVE_DIR = os.path.join('Results/', f"{DATA_CLEANLINESS}{args.dataset_name}")
    FIGURES_SAVE_DIR = os.path.join('Figures/', f'{DATA_CLEANLINESS}{args.dataset_name}')
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    os.makedirs(FIGURES_SAVE_DIR, exist_ok=True)
    SEED = 42
    HARDNESS_SAVE_DIR = f"Results/{DATA_CLEANLINESS}{args.dataset_name}/"

    THRESHOLDS = np.arange(10, 100, 10)

    main()
