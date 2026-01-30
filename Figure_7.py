"""This module performs the experiments to create the Figure 7 of the paper."""


import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from config import DEVICE, get_config, ROOT
from data import load_dataset
from neural_networks import ResNet18LowRes


def evaluate_model_per_class_precision_recall(model, dataloader, num_classes):
    """Compute per-class precision and recall for a single model."""
    tp = torch.zeros(num_classes, device=DEVICE)
    fp = torch.zeros(num_classes, device=DEVICE)
    fn = torch.zeros(num_classes, device=DEVICE)

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

            for c in range(num_classes):
                tp[c] += ((predictions == c) & (targets == c)).sum()
                fp[c] += ((predictions == c) & (targets != c)).sum()
                fn[c] += ((predictions != c) & (targets == c)).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    precision *= 100
    recall *= 100

    return precision.cpu().numpy(), recall.cpu().numpy()


def main():
    """Main function that runs the code."""
    for dataset_name in ['CIFAR10', 'CIFAR100']:
        config = get_config(dataset_name)
        num_classes = config['num_classes']
        num_epochs = config['num_epochs']
        model_dir = config['save_dir']

        _, _, test_loader, _ = load_dataset(dataset_name)

        all_precisions, all_recalls = [], []

        for model_id in range(5):
            model_path = os.path.join(model_dir, 'none', f'unclean{dataset_name}',
                                      f'dataset_0_model_{model_id}_epoch_{num_epochs}.pth')

            model = ResNet18LowRes(num_classes=num_classes).to(DEVICE)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            precision, recall = evaluate_model_per_class_precision_recall(model, test_loader, num_classes)

            all_precisions.append(precision)
            all_recalls.append(recall)

        # Shape: (num_models, num_classes)
        all_precisions = np.stack(all_precisions)
        all_recalls = np.stack(all_recalls)

        mean_precision = all_precisions.mean(axis=0)
        mean_recall = all_recalls.mean(axis=0)

        # Sort classes by mean recall (same logic as Figure 1)
        sorted_indices = np.argsort(mean_recall)

        sorted_precision = mean_precision[sorted_indices]
        sorted_recall = mean_recall[sorted_indices]

        # Correlation analysis
        pearson_corr, pearson_p = pearsonr(sorted_precision, sorted_recall)
        spearman_corr, spearman_p = spearmanr(sorted_precision, sorted_recall)

        print(f"\n{dataset_name}")
        print(f"Pearson correlation (precision vs recall):  {pearson_corr:.4f} (p={pearson_p:.4e})")
        print(f"Spearman correlation (precision vs recall): {spearman_corr:.4f} (p={spearman_p:.4e})")

        # Plot
        plt.figure(figsize=(8, 5))
        x = np.arange(num_classes)

        plt.plot(x, sorted_recall, marker='o', linewidth=2, label='Recall')
        plt.plot(x, sorted_precision, marker='s', linewidth=2, label='Precision')

        plt.xlabel("Class (sorted by mean recall)")
        plt.ylabel("Score (%)")
        plt.title(f"Per-Class Precision & Recall ({dataset_name})")
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(ROOT, 'Figures/', f"unclean{dataset_name}", "Figure_precision_recall.pdf")
        plt.savefig(fig_path)


if __name__ == '__main__':
    main()
