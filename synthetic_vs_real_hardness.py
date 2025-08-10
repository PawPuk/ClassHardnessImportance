import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torchvision
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, TensorDataset
import tqdm

from config import get_config, ROOT
from data import AugmentedSubset, load_dataset, IndexedDataset
from neural_networks import ResNet18LowRes
from utils import set_reproducibility


def generate_smote_dataset(dataset, num_classes, num_samples, n_neighbors=5):
    images_list, labels_list, synthetic_samples, synthetic_labels = [], [], [], []
    for img, label, _ in dataset:
        images_list.append(img.view(-1).numpy())  # flatten to vector
        labels_list.append(label)
    images_list = np.stack(images_list)
    labels_list = np.array(labels_list)

    print("Running SMOTE...")

    for cls in tqdm.tqdm(range(num_classes), desc='Generating synthetic data via SMOTE.'):
        class_mask = (labels_list == cls)
        X_cls = images_list[class_mask]
        # Fit neighbors on current class
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_cls)
        neigh_indices = nn.kneighbors(X_cls, return_distance=False)
        for _ in range(num_samples[cls]):
            idx = np.random.randint(0, X_cls.shape[0])  # TODO: Check if .shape accomplishes this goal!!!
            x = X_cls[idx]
            # Choose one of its k neighbors (not itself)
            neighbor_idx = np.random.choice(neigh_indices[idx][1:])
            x_neighbor = X_cls[neighbor_idx]
            # Interpolate
            lam = np.random.rand()
            x_new = x + lam * (x_neighbor - x)
            synthetic_samples.append(x_new)
            synthetic_labels.append(cls)

    print(f"Generated {len(synthetic_samples)} synthetic samples using custom SMOTE.")
    # Convert to tensor format
    all_X = np.stack(synthetic_samples)
    all_y = np.array(synthetic_labels)
    all_X = torch.tensor(all_X, dtype=torch.float32).view(-1, 3, 32, 32)
    all_y = torch.tensor(all_y, dtype=torch.long)
    return IndexedDataset(TensorDataset(all_X, all_y)), all_y


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
    os.makedirs(full_dataset_dir, exist_ok=True)

    for file in os.listdir(full_dataset_dir):
        if file.endswith(".pth") and f"_epoch_{config['num_epochs']}" in file:
            model_path = os.path.join(full_dataset_dir, file)
            model_state = torch.load(model_path)
            model_states.append(model_state)

    print(f"Loaded {len(model_states)} models for estimating confidence.")
    return model_states


def compute_dataset_confidences(dataloader_idx, dataloader, device, model_states, num_classes, dataset_name):
    dataset_confidences = []

    for images, labels, _ in tqdm.tqdm(dataloader, desc=f'Iterating through images in DataLoader {dataloader_idx}.'):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)
        batch_conf_sum = torch.zeros(batch_size, device=device)
        if len(model_states) > 0:
            models = [ResNet18LowRes(num_classes).to(device) for _ in model_states]
            models = [model.load_state_dict(model_states[i]) for i, model in enumerate(models)]
        else:
            models = [torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar{dataset_name}_resnet{i}",
                                     pretrained=True)
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


def visualize_edm_sample_hardness(dataset_name, synthetic_dataset, synthetic_confidences, test_labels,
                                  test_confidences, num_classes, class_names):
    # Compute per-class average confidence from test set
    test_class_means = [test_confidences[test_labels == c].mean().item() for c in range(num_classes)]

    # Print top-20% of the hardest classes
    class_avg_confidences = torch.tensor(test_class_means)
    sorted_class_ids = torch.argsort(class_avg_confidences)

    top_k = max(1, num_classes // 5)
    hardest_classes = sorted_class_ids[:top_k].tolist()

    print(f"\nTop {top_k} hardest classes (lowest avg test confidence):")
    for rank, class_id in enumerate(hardest_classes, 1):
        conf = class_avg_confidences[class_id].item()
        print(f"{rank}. Class {class_id} — Avg Test Confidence: {conf:.4f}")

    # Extract EDM image tensors and labels
    edm_images, edm_labels = [], []
    for img, label, _ in synthetic_dataset:
        edm_images.append(img)
        edm_labels.append(label)
    edm_images = torch.stack(edm_images)
    edm_labels = torch.tensor(edm_labels)

    output_dir = os.path.join(ROOT, 'Figures', dataset_name, 'EDM_Hardness')
    os.makedirs(output_dir, exist_ok=True)

    for c in range(num_classes):
        cls_mask = edm_labels == c
        cls_imgs = edm_images[cls_mask]
        cls_confs = synthetic_confidences[cls_mask]
        test_avg = test_class_means[c]

        # Hardest 20
        hardest_idx = torch.argsort(cls_confs)[:20]
        hardest_imgs = cls_imgs[hardest_idx]

        # 20 closest to test average
        diff = torch.abs(cls_confs - test_avg)
        closest_idx = torch.argsort(diff)[:20]
        closest_imgs = cls_imgs[closest_idx]

        def plot_and_save(imgs, title, fname):
            grid = make_grid(imgs, nrow=5, normalize=True)
            plt.figure(figsize=(10, 6))
            plt.imshow(grid.permute(1, 2, 0).cpu())
            plt.axis('off')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"class_{c}_{fname}.png"))
            plt.close()

        plot_and_save(hardest_imgs, f"{class_names[c].capitalize()} - 20 Hardest EDM Samples", "hardest")

        plot_and_save(closest_imgs, f"{class_names[c].capitalize()} - 20 Avg-like EDM Samples", "avg_like")


def compute_pruned_class_level_confidence(confidences, labels, remaining_count, num_classes, num_trials=10,
                                          k_percentiles=(1.0, 0.5, 0.25, 0.1)):
    assert confidences.shape == labels.shape
    k_percentiles = sorted(k_percentiles, reverse=True)  # ensure 1.0 (100%) is first
    all_k_results = {k: [] for k in k_percentiles}
    all_k_labels = {k: [] for k in k_percentiles}
    for _ in range(num_trials):
        remaining_confidences, remaining_labels = [], []
        for cls_idx in range(num_classes):
            cls_indices = np.where(labels == cls_idx)[0]
            n_to_keep = remaining_count[cls_idx]
            selected_indices = np.random.choice(cls_indices, size=n_to_keep, replace=False)
            remaining_confidences.append(confidences[selected_indices])
            remaining_labels.append(np.full(n_to_keep, cls_idx))
        remaining_confidences = np.concatenate(remaining_confidences)
        remaining_labels = np.concatenate(remaining_labels)

        for k in k_percentiles:
            k_confidences, k_labels = [], []
            for cls_idx in range(num_classes):
                cls_conf = remaining_confidences[remaining_labels == cls_idx]
                n_k = max(1, int(len(cls_conf) * k))  # at least one sample
                hardest_indices = np.argsort(cls_conf)[:n_k]  # bottom-k (hardest)
                k_confidences.append(cls_conf[hardest_indices])
                k_labels.append(np.full(n_k, cls_idx))
            k_confidences = np.concatenate(k_confidences)
            k_labels = np.concatenate(k_labels)
            # Compute class-wise average confidence
            class_avg_conf = np.zeros(num_classes)
            for cls_idx in range(num_classes):
                cls_conf = k_confidences[k_labels == cls_idx]
                class_avg_conf[cls_idx] = cls_conf.mean()
            all_k_results[k].append(class_avg_conf)
            all_k_labels[k].append(k_labels)
    mean_class_conf, std_class_conf = {}, {}
    for k in k_percentiles:
        trials = np.stack(all_k_results[k])  # shape: (num_trials, num_classes)
        mean_class_conf[k] = np.nanmean(trials, axis=0)
        std_class_conf[k] = np.nanstd(trials, axis=0)
    return mean_class_conf, std_class_conf


def plot_class_confidences(base_index, results, dataset_name, num_classes, sort_idx):
    plt.figure(figsize=(14, 7))
    color_map = ['blue', 'red', 'green', 'orange', 'black']
    line_styles = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 5, 1, 5))]

    for i, (dataloader_name, (mean_class_conf, std_class_conf), _) in enumerate(results):
        x = np.arange(num_classes)
        for j, k in enumerate(sorted(mean_class_conf.keys(), reverse=True)):  # plot larger subsets first
            means_k = np.array(mean_class_conf[k])[sort_idx]
            stds_k = np.array(std_class_conf[k])[sort_idx]

            plt.plot(x, means_k, label=f'{dataloader_name} {int(k*100)}%', color=color_map[i],
                     linestyle=line_styles[j % len(line_styles)], linewidth=2, alpha=0.9)

            plt.fill_between(x, means_k - stds_k, means_k + stds_k,
                             color=color_map[i], alpha=0.15, linewidth=0)

    plt.title(f"Class-level confidences (sorted by {results[base_index][0]})")
    plt.xlabel("Sorted class index")
    plt.ylabel("Mean confidence ± std")
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    save_dir = os.path.join(ROOT, 'Figures', dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{results[base_index][0]}_confidence.pdf"), bbox_inches='tight')
    plt.close()


def main(dataset_name: str):
    config = get_config(dataset_name)
    num_classes = config['num_classes']
    class_names = config['class_names']
    num_training_samples = config['num_training_samples']
    num_test_samples = config['num_test_samples']

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
    print('Loaded synthetic data generated through EDM.')

    samples_per_class = [np.sum(synthetic_labels == c) for c in range(num_classes)]
    training_smote_dataset, training_smote_labels = generate_smote_dataset(training_dataset, num_classes,
                                                                           samples_per_class)
    test_smote_dataset, test_smote_labels = generate_smote_dataset(test_dataset, num_classes, samples_per_class)

    dataloader_names = ['Training', 'Training_SMOTE', 'Test', 'Test_SMOTE', 'EDM']
    loaders = [DataLoader(dataset, batch_size=5000, shuffle=False)
               for dataset in [training_dataset, training_smote_dataset, test_dataset, test_smote_dataset,
                               synthetic_dataset]]

    model_states = load_model_states(dataset_name, config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_labels = torch.tensor([label for _, label, _ in training_dataset])
    test_labels = torch.tensor([label for _, label, _ in test_dataset])
    all_average_confidences = []

    for dataloader_idx, dataloader in enumerate(loaders):
        dataloader_name = dataloader_names[dataloader_idx]
        results_path = os.path.join(ROOT, f'Results', dataset_name)
        os.makedirs(results_path, exist_ok=True)
        confidences_path = os.path.join(results_path, f'{dataloader_name}_confidences.pt')
        if os.path.exists(confidences_path):
            dataset_confidences = torch.load(confidences_path)
            print(f'Loaded {dataloader_name} DataLoader.')
        else:
            dataset_confidences = compute_dataset_confidences(dataloader_idx, dataloader, device, model_states,
                                                              num_classes, 10 if dataloader_name == 'CIFAR10' else 100)
            torch.save(dataset_confidences, confidences_path)

        labels = [training_labels, training_smote_labels, test_labels, test_smote_labels,
                  torch.from_numpy(synthetic_labels)]
        all_average_confidences.append([dataloader_name, dataset_confidences, labels[dataloader_idx]])

    visualize_edm_sample_hardness(dataset_name, synthetic_dataset, all_average_confidences[4][1], test_labels,
                                  all_average_confidences[2][1], num_classes, class_names)

    backup_all_average_confidences = copy.deepcopy(all_average_confidences)

    for i in range(4):
        num_samples = num_training_samples if i < 2 else num_test_samples
        all_average_confidences[i][1] = compute_pruned_class_level_confidence(backup_all_average_confidences[i][1],
                                                                              backup_all_average_confidences[i][2],
                                                                              num_samples, num_classes)
    all_average_confidences[4][1] = compute_pruned_class_level_confidence(backup_all_average_confidences[4][1],
                                                                          backup_all_average_confidences[4][2],
                                                                          num_training_samples, num_classes)
    sort_idx = np.argsort(all_average_confidences[4][1][0][1.0])  # Sort by full 100% confidence
    for i in range(2):
        plot_class_confidences(0, [all_average_confidences[i]] + [all_average_confidences[4]], dataset_name,
                               num_classes, sort_idx)

    all_average_confidences[4][1] = compute_pruned_class_level_confidence(backup_all_average_confidences[4][1],
                                                                          backup_all_average_confidences[4][2],
                                                                          num_test_samples, num_classes)
    for i in range(2, 4):
        plot_class_confidences(0, [all_average_confidences[i]] + [all_average_confidences[4]], dataset_name,
                               num_classes, sort_idx)


if __name__ == '__main__':
    set_reproducibility()

    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-10 or CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR10', 'CIFAR100'], help='Dataset name: CIFAR10 or CIFAR100')

    args = parser.parse_args()

    main(args.dataset_name)


"""TODOs
  - Rerun the experiments using the pre-trained models I have on HPC.
"""