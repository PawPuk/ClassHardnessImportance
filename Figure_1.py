"""This module performs the experiments to create the Figure 1 of the paper."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import DEVICE, get_config, ROOT
from data import load_dataset
from neural_networks import ResNet18LowRes


def evaluate_model_per_class(model, dataloader, num_classes):
    """Compute per-class recall for a single model."""
    correct = torch.zeros(num_classes, device=DEVICE)
    total = torch.zeros(num_classes, device=DEVICE)

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

            for c in range(num_classes):
                mask = (targets == c)
                total[c] += mask.sum()
                correct[c] += (predictions[mask] == c).sum()

    per_class_rec = correct / total * 100
    return per_class_rec.cpu().numpy()


def main():
    """Main function that runs the code."""
    for dataset_name in ['CIFAR10', 'CIFAR100']:
        config = get_config(dataset_name)
        num_classes = config['num_classes']
        num_epochs = config['num_epochs']
        model_dir = config['save_dir']
        class_labels = config['class_names']

        _, _, test_loader, _ = load_dataset(dataset_name)

        all_model_recalls = []
        for model_id in range(5):
            model_path = os.path.join(model_dir, 'none', f'unclean{dataset_name}',
                                      f'dataset_0_model_{model_id}_epoch_{num_epochs}.pth')

            model = ResNet18LowRes(num_classes=num_classes).to(DEVICE)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            per_class_rec = evaluate_model_per_class(model, test_loader, num_classes)
            all_model_recalls.append(per_class_rec)

        # Shape: (num_models, num_classes)
        all_model_recalls = np.stack(all_model_recalls)

        mean_rec = all_model_recalls.mean(axis=0)
        std_rec = all_model_recalls.std(axis=0)
        min_rec = all_model_recalls.min(axis=0, initial=100)
        max_rec = all_model_recalls.max(axis=0, initial=0)

        # Identify hardest and easiest classes (bottom/top 20%)
        k = max(1, int(0.2 * num_classes))
        sorted_indices = np.argsort(mean_rec)

        sorted_mean_rec = mean_rec[sorted_indices]
        sorted_std_rec = std_rec[sorted_indices]
        sorted_min_rec = min_rec[sorted_indices]
        sorted_max_rec = max_rec[sorted_indices]

        hardest_classes = sorted_indices[:k]
        easiest_classes = sorted_indices[-k:]

        print(f"\nHardest {k} classes (lowest mean recall):")
        for c in hardest_classes:
            print(f"  Class {c} ({class_labels[c]}): {mean_rec[c]:.4f}")

        print(f"\nEasiest {k} classes (highest mean recall):")
        for c in easiest_classes:
            print(f"  Class {c} ({class_labels[c]}): {mean_rec[c]:.4f}")

        # Plot bar chart
        plt.figure(figsize=(8, 5))
        x = np.arange(num_classes)

        plt.vlines(x, sorted_min_rec, sorted_max_rec, color="black", linewidth=0.8, alpha=0.6)
        plt.vlines(x, sorted_mean_rec - sorted_std_rec, sorted_mean_rec + sorted_std_rec, linewidth=2, alpha=0.9)
        plt.scatter(x, sorted_mean_rec, s=20)
        global_mean = mean_rec.mean()
        plt.axhline(global_mean, linestyle='--', color='gray', linewidth=1, alpha=0.8)

        plt.xlabel("Class (sorted by mean recall)")
        plt.ylabel("Recall")
        plt.title(f"Mean Recall ({dataset_name})")
        plt.tight_layout()

        fig_path = os.path.join(ROOT, 'Figures/', f"unclean{dataset_name}", "Figure_1.pdf")
        plt.savefig(fig_path)


if __name__ == '__main__':
    main()
