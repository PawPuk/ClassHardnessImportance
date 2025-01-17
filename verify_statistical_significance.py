import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_rel
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


def collect_el2n_scores(loader, n):
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
        for i in range(n):
            all_el2n_scores.append(el2n_scores)
        model_class_accuracies.append(class_accuracies)
    return all_el2n_scores, model_class_accuracies


def save_el2n_scores(el2n_scores):
    with open(os.path.join(RESULTS_SAVE_DIR, 'el2n_scores.pkl'), 'wb') as file:
        pickle.dump(el2n_scores, file)


def load_results(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def compute_pruned_indices_and_measure_statistical_significance(el2n_scores, aum_scores, forgetting_scores):
    num_models = len(el2n_scores[0])  # number of models in the ensemble
    thresholds = np.arange(10, 100, 10)
    results, statistical_significance_of_hardness_estimators = {}, {}

    for metric_name, metric_scores in tqdm([("el2n", el2n_scores), ("aum", aum_scores),
                                            ("forgetting", forgetting_scores)], desc='Iterating through metrics.'):
        metric_scores = np.array(metric_scores)
        results[metric_name] = []
        statistical_significance_of_hardness_estimators[metric_name] = []
        for thresh in thresholds:
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

                if num_ensemble_models == num_models:
                    prev_avg_hardness_scores = np.mean(metric_scores[:num_ensemble_models-1], axis=0)
                    t_stat, p_value = ttest_rel(prev_avg_hardness_scores, avg_hardness_scores)
                    statistical_significance_of_hardness_estimators[metric_name].append((t_stat, p_value))
            results[metric_name].append(metric_results)
            print(f'For metric {metric_name} and threshold {thresh} we are pruning {len(metric_results[0])} samples,'
                  f' and loop through of ensemble of various sizes up to {len(metric_results)}.')
    return results, statistical_significance_of_hardness_estimators


def visualize_statistical_significance(statistical_significance_of_hardness_estimators):
    thresholds = np.arange(10, 100, 10)

    metrics = list(statistical_significance_of_hardness_estimators.keys())
    t_stats = {metric: [t_stat for t_stat, _ in statistical_significance_of_hardness_estimators[metric]] for metric in
               metrics}
    p_values = {metric: [p_value for _, p_value in statistical_significance_of_hardness_estimators[metric]] for metric
                in metrics}

    # Plotting T-Statistics
    plt.figure(figsize=(10, 6))
    for metric, t_stat_values in t_stats.items():
        plt.plot(thresholds, t_stat_values, label=f'{metric.upper()}', marker='o')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, label='No Effect (t=0)')
    plt.title('T-Statistics Across Thresholds for Hardness Estimators')
    plt.xlabel('Threshold (%)')
    plt.ylabel('T-Statistic')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(FIGURES_SAVE_DIR, f'statistical_significance_of_hardness_estimator.pdf'))

    # Plotting P-Values
    plt.figure(figsize=(10, 6))
    for metric, p_value_values in p_values.items():
        plt.plot(thresholds, p_value_values, label=f'{metric.upper()}', marker='o')
    plt.axhline(0.05, color='red', linestyle='--', linewidth=1, label='Significance Level (p=0.05)')
    plt.title('P-Values Across Thresholds for Hardness Estimators')
    plt.xlabel('Threshold (%)')
    plt.ylabel('P-Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(FIGURES_SAVE_DIR, f'p_values_of_hardness_estimator.pdf'))


def compute_and_visualize_stability_of_pruning(results):
    metrics = ['el2n', 'aum', 'forgetting']

    for metric in metrics:
        metric_results = results[metric]
        num_thresholds = len(metric_results)
        num_models = len(metric_results[0]) - 1
        stability_results = np.zeros((num_thresholds, num_models))

        for i in range(num_thresholds):  # Loop over thresholds (rows)
            for j in range(num_models):  # Loop over model pairs (columns)
                print(len(metric_results[i]), len(metric_results[i][j]))
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


def main():
    training_loader, training_set_size, _, _ = load_dataset(args.dataset_name, args.remove_noise, SEED, False)
    training_all_el2n_scores, training_class_accuracies = collect_el2n_scores(training_loader, training_set_size)
    save_el2n_scores((training_all_el2n_scores, training_class_accuracies))

    aum_path = os.path.join(HARDNESS_SAVE_DIR, 'AUM.pkl')
    forgetting_path = os.path.join(HARDNESS_SAVE_DIR, 'Forgetting.pkl')
    AUM_over_epochs_and_models = load_results(aum_path)
    AUM_over_epochs = [
        [
            sum(model_list[sample_idx][epoch_idx] for epoch_idx in range(NUM_EPOCHS)) / len(NUM_EPOCHS)
            for sample_idx in range(NUM_SAMPLES)
        ]
        for model_list in AUM_over_epochs_and_models
    ]
    AUM_scores = np.mean(AUM_over_epochs, axis=1)

    forgetting_scores = load_results(forgetting_path)
    print(len(AUM_scores), len(AUM_scores[0]))
    print(len(forgetting_scores), len(forgetting_scores[0]))
    print(len(training_all_el2n_scores), len(training_all_el2n_scores[0]))

    pruned_indices, hardness_ss = compute_pruned_indices_and_measure_statistical_significance(
        training_all_el2n_scores, AUM_scores, forgetting_scores)
    visualize_statistical_significance(hardness_ss)
    compute_and_visualize_stability_of_pruning(pruned_indices)


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
    SEED = 42
    HARDNESS_SAVE_DIR = f"Results/{DATA_CLEANLINESS}{args.dataset_name}/"

    main()
