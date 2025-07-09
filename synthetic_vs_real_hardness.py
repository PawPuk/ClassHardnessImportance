import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import tqdm

from config import get_config, ROOT
from data import AugmentedSubset, load_dataset, IndexedDataset
from neural_networks import ResNet18LowRes
from utils import set_reproducibility


def convert_numpy_to_dataset(images_np, labels_np, config):
    # TODO: This /255.0 is sketchy
    images = torch.from_numpy(images_np).float() / 255.0  # from uint8 to float32
    labels = torch.from_numpy(labels_np).long()

    # Apply normalization batch-wise
    images = images.permute(0, 3, 1, 2)  # HWC -> CHW
    images = torchvision.transforms.Normalize(config['mean'], config['std'])(images)

    return IndexedDataset(TensorDataset(images, labels))


def load_model_states(dataset_name, config):
    models_dir = os.path.join(ROOT, "Models")
    model_states = []
    full_dataset_dir = os.path.join(models_dir, "none", f"unclean{dataset_name}")

    for file in os.listdir(full_dataset_dir):
        if file.endswith(".pth") and f"_epoch_{config['num_epochs']}" in file:
            model_path = os.path.join(full_dataset_dir, file)
            model_state = torch.load(model_path)
            model_states.append(model_state)

    print(f"Loaded {len(model_states)} models for estimating confidence.")
    return model_states


def compute_dataset_confidences(dataloader_idx, dataloader, device, model_states, num_classes):
    dataset_confidences = []

    for images, labels, _ in tqdm.tqdm(dataloader, desc=f'Iterating through images in DataLoader {dataloader_idx}.'):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)
        batch_conf_sum = torch.zeros(batch_size, device=device)
        if len(model_states) > 0:
            models = [ResNet18LowRes(num_classes).load_state_dict(model_state) for model_state in model_states]
        else:
            models = [torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar10_resnet{i}", pretrained=True)
                      for i in [20, 32, 44]]
        for model in models:
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
                correct_confs = softmax_outputs[torch.arange(batch_size), labels]
                batch_conf_sum += correct_confs
            del model
        avg_conf = batch_conf_sum / len(models)
        dataset_confidences.append(avg_conf.cpu())
    dataset_confidences = torch.cat(dataset_confidences)
    print(f"Finished dataloader {dataloader_idx}, total samples: {dataset_confidences.shape[0]}")
    return dataset_confidences


def plot_class_confidences(base_indices, results, dataset_name, num_classes):
    for base_dataloader_idx in base_indices:
        base_conf = results[base_dataloader_idx][1]
        base_labels = results[base_dataloader_idx][2]

        # Compute class-level means
        class_means = []
        for c in range(num_classes):
            class_mask = (base_labels == c)
            class_mean = base_conf[class_mask].mean().item()
            class_means.append(class_mean)

        class_means = np.array(class_means)
        sort_idx = np.argsort(class_means)

        plt.figure(figsize=(12, 6))
        for i, (dataloader_name, confidences, labels_for_data) in enumerate(results):
            per_class_means, per_class_stds = [], []
            for c in range(num_classes):
                mask = (labels_for_data == c)
                if i == 2:
                    print(f"Class {c} contains {torch.sum(mask).item()} data samples.")
                conf_vals = confidences[mask].numpy()
                per_class_means.append(conf_vals.mean())
                per_class_stds.append(conf_vals.std())

            per_class_means = np.array(per_class_means)[sort_idx]
            per_class_stds = np.array(per_class_stds)[sort_idx]
            x = np.arange(len(per_class_means))

            plt.plot(x, per_class_means, label=f'{dataloader_name}_confidence')
            plt.fill_between(x, per_class_means - per_class_stds, per_class_means + per_class_stds, alpha=0.2)

        plt.title(f"Class-level confidences sorted by {results[base_dataloader_idx][0]}")
        plt.xlabel("Sorted classes")
        plt.ylabel("Confidence")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(ROOT, 'Figures', dataset_name, f"{results[base_dataloader_idx][0]}_confidence.pdf"))


def main(dataset_name: str):
    config = get_config(dataset_name)
    num_classes = config['num_classes']

    _, training_dataset, _, test_dataset = load_dataset(dataset_name, False, False, False)
    new_training_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(config['mean'], config['std']),
    ])
    training_dataset = AugmentedSubset(training_dataset, transform=new_training_transform)

    synthetic_data = np.load(os.path.join(ROOT, f'GeneratedImages/{dataset_name}.npz'))
    image_key, label_key = synthetic_data.files
    synthetic_images = synthetic_data[image_key]
    synthetic_labels = synthetic_data[label_key]
    synthetic_dataset = convert_numpy_to_dataset(synthetic_images, synthetic_labels, config)

    dataloader_names = ['Training', 'Test', 'EDM']
    loaders = [DataLoader(dataset, batch_size=5000, shuffle=False) for dataset in [training_dataset, test_dataset,
                                                                                   synthetic_dataset]]

    model_states = load_model_states(dataset_name, config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_labels = torch.tensor([label for _, label, _ in training_dataset])
    test_labels = torch.tensor([label for _, label, _ in test_dataset])
    all_average_confidences = []

    for dataloader_idx, dataloader in enumerate(loaders):
        dataloader_name = dataloader_names[dataloader_idx]
        confidences_path = os.path.join(ROOT, f'Results', dataset_name, f'{dataloader_name}_confidences.pt')
        if os.path.exists(confidences_path):
            dataset_confidences = torch.load(confidences_path)
        else:
            dataset_confidences = compute_dataset_confidences(dataloader_idx, dataloader, device, model_states,
                                                              num_classes)
            torch.save(dataset_confidences, confidences_path)

        labels = [train_labels, test_labels, torch.from_numpy(synthetic_labels)]
        all_average_confidences.append((dataloader_name, dataset_confidences, labels[dataloader_idx]))

    base_dataloaders_indices = [0]
    plot_class_confidences(base_dataloaders_indices, all_average_confidences, dataset_name, num_classes)


if __name__ == '__main__':
    set_reproducibility()

    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-10 or CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR10', 'CIFAR100'], help='Dataset name: CIFAR10 or CIFAR100')

    args = parser.parse_args()

    main(args.dataset_name)
