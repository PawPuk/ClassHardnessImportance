"""This module was used to find the sample and corresponding probabilities used for Figure 3 of the paper."""

import os
import torch
import matplotlib.pyplot as plt

from config import get_config, DEVICE, ROOT
from data import load_dataset
from neural_networks import ResNet18LowRes


def main():
    """Main function that runs the code."""
    dataset_name = 'CIFAR10'
    config = get_config(dataset_name)
    model_dir = config['save_dir']
    class_labels = config['class_names']
    num_epochs = config['num_epochs']

    training_loader, _, _, _ = load_dataset(dataset_name)

    model_path = os.path.join(model_dir, 'none', 'uncleanCIFAR10', f'dataset_0_model_0_epoch_{num_epochs}.pth')

    model = ResNet18LowRes(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    output_dir = os.path.join(ROOT, "Figures", "high_conf_low_margin")
    os.makedirs(output_dir, exist_ok=True)

    samples = []
    global_index = 0

    with torch.no_grad():
        for images, labels, _ in training_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            for i in range(images.size(0)):
                p = probs[i].cpu()
                image = images[i].cpu()
                label = labels[i].cpu().item()

                sorted_p, _ = torch.sort(p, descending=True)
                confidence = sorted_p[0].item()
                margin = (sorted_p[0] - sorted_p[1]).item()
                score = margin - confidence

                samples.append({
                    "index": global_index,
                    "image": image,
                    "label": label,
                    "probs": p,
                    "confidence": confidence,
                    "margin": margin,
                    "score": score
                })

                global_index += 1

    # Select top 10 by (confidence - margin)
    samples.sort(key=lambda x: x["score"], reverse=True)
    top_samples = samples[:10]

    for rank, s in enumerate(top_samples):
        img = s["image"].permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img)
        ax.axis("off")

        header = (
            f"Rank: {rank}\n"
            f"Index: {s['index']}\n"
            f"Label: {class_labels[s['label']]}\n"
            f"Confidence: {s['confidence']:.4f}\n"
            f"Margin: {s['margin']:.4f}\n"
        )

        prob_text = "\n".join(f"{class_labels[i]}: {s['probs'][i]:.4f}" for i in range(len(class_labels)))

        ax.set_title(header + prob_text, fontsize=9)

        save_path = os.path.join(output_dir, f"sample_rank_{rank}_idx_{s['index']}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    main()
