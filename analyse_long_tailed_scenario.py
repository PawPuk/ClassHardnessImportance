import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import utils as u


def compute_sample_hardness(model, dataloader):
    """Computes the hardness of each sample based on the confidence of the model."""
    model.eval()
    confidences_and_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(u.DEVICE)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confidences, _ = torch.max(probs, dim=1)  # Confidence for predicted class
            confidences_and_labels.extend(list(zip(confidences.cpu().numpy(), labels.cpu().numpy())))
    return confidences_and_labels


def prune_dataset(confidences_and_labels, prune_percentage=50):
    """Prunes the dataset by removing the easiest samples based on confidence."""
    prune_threshold = prune_percentage / 100

    # Sort by confidence (lowest confidence comes first)
    sorted_confidences_and_labels = sorted(confidences_and_labels, key=lambda x: x[0])

    # Remove the bottom prune_threshold% samples
    num_samples_to_prune = int(len(sorted_confidences_and_labels) * prune_threshold)
    pruned_confidences_and_labels = sorted_confidences_and_labels[num_samples_to_prune:]

    # Extract the remaining labels after pruning
    remaining_labels = [label for _, label in pruned_confidences_and_labels]

    return remaining_labels


def compute_class_distribution(remaining_labels, num_classes=100):
    """Computes the distribution of remaining samples across classes."""
    class_counts = np.zeros(num_classes, dtype=int)

    for label in remaining_labels:
        class_counts[label] += 1

    return class_counts


def process_models_and_prune_samples(train_loader, prune_percentage=50):
    """Loads each model, computes sample hardness, prunes dataset, and computes class distribution."""
    class_distributions_all_models = []
    model_files = sorted(
        [f for f in os.listdir(u.MODELS_DIR) if f.endswith('.pth')])  # Assuming models are saved as .pth files

    for model_idx, model_file in enumerate(tqdm(model_files)):
        print(f"Processing model: {model_file}")

        # Load the model
        model_path = os.path.join(u.MODELS_DIR, model_file)
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_0", pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=u.DEVICE))
        model.to(u.DEVICE)

        # Compute confidence (hardness) for all samples
        confidences_and_labels = compute_sample_hardness(model, train_loader)

        # Prune the dataset (remove 50% of the easiest samples)
        remaining_labels = prune_dataset(confidences_and_labels, prune_percentage)

        # Compute the class distribution after pruning
        class_distribution = compute_class_distribution(remaining_labels)
        class_distributions_all_models.append(class_distribution)

        # Plot long-tailed distribution for this model
        plot_long_tailed_distribution(class_distribution, model_idx, prune_percentage)

        # Free up memory by deleting the model and clearing GPU cache
        del model
        torch.cuda.empty_cache()

    return class_distributions_all_models


def plot_long_tailed_distribution(class_distribution, model_idx, prune_percentage):
    """Plots the long-tailed class distribution after pruning for each model."""
    # Sort the class distribution by the number of samples (in ascending order)
    sorted_class_distribution = sorted(class_distribution)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_class_distribution)), sorted_class_distribution, color='b', edgecolor='black')
    plt.xlabel('Sorted Class Index (by number of samples)')
    plt.ylabel('Number of Samples')
    plt.title(f'Class Distribution After {prune_percentage}% Pruning (Model {model_idx + 1})')

    # Save the plot
    plt.savefig(f'{u.DISTRIBUTION_FIGURES_DIR}long_tailed_distribution_model_{model_idx + 1}.pdf')
    plt.show()


def compute_consecutive_distributions_correlations(distributions_per_model):
    """Computes Pearson and Spearman correlations between consecutive models' class distributions."""
    pearson_correlations, spearman_correlations, pearson_p_values, spearman_p_values = [], [], [], []

    for i in range(len(distributions_per_model) - 1):
        dist_1, dist_2 = distributions_per_model[i], distributions_per_model[i + 1]

        # Compute Pearson correlation and p-value
        pearson_corr, pearson_p = pearsonr(dist_1, dist_2)
        pearson_correlations.append(pearson_corr)
        pearson_p_values.append(pearson_p)

        # Compute Spearman correlation and p-value
        spearman_corr, spearman_p = spearmanr(dist_1, dist_2)
        spearman_correlations.append(spearman_corr)
        spearman_p_values.append(spearman_p)

    return pearson_correlations, spearman_correlations, pearson_p_values, spearman_p_values


def compute_last_model_distributions_correlations(distributions_per_model):
    """Computes Pearson and Spearman correlations between each model's distribution and the last model's distribution."""
    pearson_correlations, spearman_correlations, pearson_p_values, spearman_p_values = [], [], [], []
    last_model_dist = distributions_per_model[-1]  # Get the distribution of the last model

    for i in range(len(distributions_per_model) - 1):
        dist_current = distributions_per_model[i]

        # Compute Pearson correlation and p-value
        pearson_corr, pearson_p = pearsonr(dist_current, last_model_dist)
        pearson_correlations.append(pearson_corr)
        pearson_p_values.append(pearson_p)

        # Compute Spearman correlation and p-value
        spearman_corr, spearman_p = spearmanr(dist_current, last_model_dist)
        spearman_correlations.append(spearman_corr)
        spearman_p_values.append(spearman_p)

    return pearson_correlations, spearman_correlations, pearson_p_values, spearman_p_values


def plot_correlations(pearson_corrs, spearman_corrs, pearson_pvals, spearman_pvals, title, filename):
    """Plots the Pearson and Spearman correlations and p-values on a single graph."""
    x = range(1, len(pearson_corrs) + 1)

    plt.figure(figsize=(10, 6))

    # Plot Pearson and Spearman correlations
    plt.plot(x, pearson_corrs, label="Pearson Correlation", marker='o', color='b', linewidth=6)
    plt.plot(x, spearman_corrs, label="Spearman Correlation", marker='x', color='r', linewidth=6)

    # Plot Pearson and Spearman p-values
    plt.plot(x, pearson_pvals, label="Pearson p-value", linestyle='--', marker='o', color='c', linewidth=6)
    plt.plot(x, spearman_pvals, label="Spearman p-value", linestyle='--', marker='x', color='m', linewidth=6)

    # Graph details
    plt.title(title)
    plt.xlabel('Model Index')
    plt.ylabel('Correlation / p-value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{u.CORRELATION_FIGURES_DIR}{filename}.pdf')
    plt.show()


if __name__ == "__main__":
    # Load the CIFAR-100 training set
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    trainset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=u.CONFIG['batch_size'], shuffle=False, num_workers=u.NUM_WORKERS)

    # Sequentially process models and prune samples for each model
    prune_percentage = 75  # You can modify this percentage
    distributions_per_model = process_models_and_prune_samples(train_loader, prune_percentage)

    # Compute and plot correlations between the distributions of consecutive models
    pearson_corrs_consec, spearman_corrs_consec, pearson_pvals_consec, spearman_pvals_consec = compute_consecutive_distributions_correlations(
        distributions_per_model)
    plot_correlations(pearson_corrs_consec, spearman_corrs_consec, pearson_pvals_consec, spearman_pvals_consec,
                      f'Correlation Between Consecutive Models\' Class Distribution After {prune_percentage}% Pruning',
                      'consecutive_distributions_correlations')

    # Compute and plot correlations between each model's distribution and the last model's distribution
    pearson_corrs_last, spearman_corrs_last, pearson_pvals_last, spearman_pvals_last = compute_last_model_distributions_correlations(
        distributions_per_model)
    plot_correlations(pearson_corrs_last, spearman_corrs_last, pearson_pvals_last, spearman_pvals_last,
                      f'Correlation Between Each Model and Last Model\'s Class Distribution After {prune_percentage}% Pruning',
                      'last_model_distributions_correlations')

