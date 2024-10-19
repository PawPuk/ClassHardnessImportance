import os

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from tqdm import tqdm

import utils as u


def compute_sample_hardness(model, dataloader):
    """Computes the hardness of each sample based on the confidence of the model."""
    model.eval()
    all_confidences = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(u.DEVICE)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confidences, _ = torch.max(probs, dim=1)  # Confidence for predicted class
            all_confidences.extend(confidences.cpu().numpy())
    return all_confidences


def process_models_and_sort_samples(train_loader):
    """Loads each model, computes sample hardness, and sorts the samples."""
    sorted_indices_all_models = []

    model_files = sorted(
        [f for f in os.listdir(u.MODELS_DIR) if f.endswith('.pth')])  # Assuming models are saved as .pth files
    for model_file in tqdm(model_files):
        print(f"Processing model: {model_file}")

        # Load the model
        model_path = os.path.join(u.MODELS_DIR, model_file)
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_0", pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=u.DEVICE))
        model.to(u.DEVICE)

        # Compute hardness for all samples in the training set
        confidences = compute_sample_hardness(model, train_loader)

        # Sort sample indices by hardness (lower confidence means harder sample)
        sorted_indices = sorted(range(len(confidences)), key=lambda k: confidences[k])

        # Save the sorted indices for this model
        sorted_indices_all_models.append(sorted_indices)

        # To reduce data complexity free up memory by deleting the model as soon as possible; also clear GPU cache.
        del model
        torch.cuda.empty_cache()

    return sorted_indices_all_models


def compute_consecutive_correlations(sorted_samples_per_model):
    """Computes Pearson and Spearman correlations between consecutive models' sorted sample indices."""
    pearson_correlations, spearman_correlations, pearson_p_values, spearman_p_values = [], [], [], []

    for i in range(len(sorted_samples_per_model) - 1):
        # Get the sorting indices for two consecutive models
        sorted_indices_1, sorted_indices_2 = sorted_samples_per_model[i], sorted_samples_per_model[i + 1]

        # Compute Pearson correlation and p-value
        pearson_corr, pearson_p = pearsonr(sorted_indices_1, sorted_indices_2)
        pearson_correlations.append(pearson_corr)
        pearson_p_values.append(pearson_p)

        # Compute Spearman correlation and p-value
        spearman_corr, spearman_p = spearmanr(sorted_indices_1, sorted_indices_2)
        spearman_correlations.append(spearman_corr)
        spearman_p_values.append(spearman_p)

    return pearson_correlations, spearman_correlations, pearson_p_values, spearman_p_values


def compute_last_model_correlations(sorted_samples_per_model):
    """Computes Pearson and Spearman correlations between each model's sorting and the last model's sorting."""
    pearson_correlations, spearman_correlations, pearson_p_values, spearman_p_values = [], [], [], []
    last_model_sorted_indices = sorted_samples_per_model[-1]  # Get the sorting of the last model

    for i in range(len(sorted_samples_per_model) - 1):
        # Get the sorting indices for the current model and the last model
        sorted_indices_current = sorted_samples_per_model[i]

        # Compute Pearson correlation and p-value
        pearson_corr, pearson_p = pearsonr(sorted_indices_current, last_model_sorted_indices)
        pearson_correlations.append(pearson_corr)
        pearson_p_values.append(pearson_p)

        # Compute Spearman correlation and p-value
        spearman_corr, spearman_p = spearmanr(sorted_indices_current, last_model_sorted_indices)
        spearman_correlations.append(spearman_corr)
        spearman_p_values.append(spearman_p)

    return pearson_correlations, spearman_correlations, pearson_p_values, spearman_p_values



def plot_correlations(pearson_corrs, spearman_corrs, pearson_pvals, spearman_pvals, title, filename):
    """Plots the Pearson and Spearman correlations and p-values on a single graph."""
    x = range(1, len(pearson_corrs) + 1)

    plt.figure(figsize=(10, 6))

    # Plot Pearson and Spearman correlations
    plt.plot(x, pearson_corrs, label="Pearson Correlation", marker='o', color='b')
    plt.plot(x, spearman_corrs, label="Spearman Correlation", marker='x', color='r')

    # Plot Pearson and Spearman p-values
    plt.plot(x, pearson_pvals, label="Pearson p-value", linestyle='--', marker='o', color='c')
    plt.plot(x, spearman_pvals, label="Spearman p-value", linestyle='--', marker='x', color='m')

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

    # Sequentially process models in the directory
    sorted_samples_per_model = process_models_and_sort_samples(train_loader)

    # Compute and plot correlations between the orderings of consecutive models
    pearson_corrs_consec, spearman_corrs_consec, pearson_pvals_consec, spearman_pvals_consec = compute_consecutive_correlations(sorted_samples_per_model)
    plot_correlations(pearson_corrs_consec, spearman_corrs_consec, pearson_pvals_consec, spearman_pvals_consec,
                      'Correlation Between Consecutive Models\' Sample Hardness Ordering', 'consecutive_correlations')

    # Compute and plot correlations between each model's sorting and the last model's sorting
    pearson_corrs_last, spearman_corrs_last, pearson_pvals_last, spearman_pvals_last = compute_last_model_correlations(sorted_samples_per_model)
    plot_correlations(pearson_corrs_last, spearman_corrs_last, pearson_pvals_last, spearman_pvals_last,
                      'Correlation Between Each Model and Last Model\'s Sample Hardness Ordering', 'last_model_correlations')


