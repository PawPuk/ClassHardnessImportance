import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torchvision
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


def plot_class_confidences(base_index, results, dataset_name, num_classes):
    base_conf = results[base_index][1]
    base_labels = results[base_index][2]

    # Compute class-level means
    class_means = []
    for c in range(num_classes):
        class_mask = (base_labels == c)
        class_mean = base_conf[class_mask].mean().item()
        class_means.append(class_mean)

    class_means = np.array(class_means)
    sort_idx = np.argsort(class_means)

    plt.figure(figsize=(12, 6))
    color_map = ['blue', 'red', 'green', 'orange', 'black']
    for i, (dataloader_name, confidences, labels_for_data) in enumerate(results):
        per_class_means, per_class_tops = [], {}
        for c in range(num_classes):
            mask = (labels_for_data == c)
            conf_vals = confidences[mask].numpy()
            if len(conf_vals) == 0:
                raise Exception  # This shouldn't happen, but is worth a sanity check.
            sorted_conf = np.sort(conf_vals)
            n = len(sorted_conf)
            ks = []
            if dataloader_name in ['Test', 'Test_SMOTE']:
                ks.append(('25%', max(1, int(0.25 * n))))
                ks.append(('50%', max(1, int(0.50 * n))))
            elif dataloader_name in ['Training', 'Training_SMOTE']:
                ks.append(('5%', max(1, int(0.05 * n))))
                ks.append(('10%', max(1, int(0.10 * n))))
            else:
                ks.append(('50%', max(1, int(0.50 * n))))
                ks.append(('25%', max(1, int(0.25 * n))))
                ks.append(('10%', max(1, int(0.10 * n))))
                ks.append(('5%', max(1, int(0.05 * n))))

            per_class_means.append(np.mean(conf_vals))
            for p, k in ks:
                if p in per_class_tops.keys():
                    per_class_tops[p].append(np.mean(sorted_conf[:k]))
                else:
                    per_class_tops[p] = [np.mean(sorted_conf[:k])]

        per_class_means = np.array(per_class_means)[sort_idx]
        for p in per_class_tops.keys():
            per_class_tops[p] = np.array(per_class_tops[p])[sort_idx]
        x = np.arange(num_classes)

        plt.plot(x, per_class_means, label=f'{dataloader_name} avg', color=color_map[i], linewidth=2)
        line_styles = ['dashed', 'dashdot', 'dotted', 'dashed', 'dashdot', 'dotted']
        if base_index == 0 or i == base_index:
            for p_idx, p in enumerate(per_class_tops.keys()):
                alpha = 0.3 + (p_idx + 1) * (0.8 - 0.3) / len(per_class_tops.keys())
                plt.plot(x, per_class_tops[p], label=f'{dataloader_name} {p}', color=color_map[i],
                         linestyle=line_styles[p_idx], alpha=alpha)

    plt.title(f"Class-level confidences sorted by {results[base_index][0]}")
    plt.xlabel("Sorted classes")
    plt.ylabel("Confidence")
    plt.legend(bbox_to_anchor=(1.22, 1.10), loc='upper right')
    plt.grid(True)

    save_dir = os.path.join(ROOT, 'Figures', dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{results[base_index][0]}_confidence.pdf"), bbox_inches='tight')
    plt.show()


def main(dataset_name: str):
    config = get_config(dataset_name)
    num_classes = config['num_classes']
    num_training_samples = config['num_training_samples']
    num_test_samples = config['num_test_samples']

    _, training_dataset, _, test_dataset = load_dataset(dataset_name, False, False, False)
    new_training_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(config['mean'], config['std']),
    ])
    training_dataset = AugmentedSubset(training_dataset, transform=new_training_transform)

    training_smote_dataset, training_smote_labels = generate_smote_dataset(training_dataset, num_classes,
                                                                           num_training_samples)
    test_smote_dataset, test_smote_labels = generate_smote_dataset(test_dataset, num_classes, num_test_samples)

    synthetic_data = np.load(os.path.join(ROOT, f'GeneratedImages/{dataset_name}.npz'))
    image_key, label_key = synthetic_data.files
    synthetic_images = synthetic_data[image_key]
    synthetic_labels = synthetic_data[label_key]
    synthetic_dataset = convert_numpy_to_dataset(synthetic_images, synthetic_labels, config)
    print('Loaded synthetic data generated through EDM.')

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
        all_average_confidences.append((dataloader_name, dataset_confidences, labels[dataloader_idx]))

    plot_class_confidences(0, all_average_confidences[:2], dataset_name, num_classes)
    plot_class_confidences(0, all_average_confidences[2:4], dataset_name, num_classes)
    plot_class_confidences(4, all_average_confidences, dataset_name, num_classes)


if __name__ == '__main__':
    set_reproducibility()

    parser = argparse.ArgumentParser(description='Train an ensemble of models on CIFAR-10 or CIFAR-100.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['CIFAR10', 'CIFAR100'], help='Dataset name: CIFAR10 or CIFAR100')

    args = parser.parse_args()

    main(args.dataset_name)


"""TODOs
  - Is convert_numpy_to_dataset correct? Are we properly transforming the EDM data?
  - Why does AugmentedSubset throw out a warning?'
  - Ensure reproducibility of the results.
  - Rerun the experiments using the pre-trained models I have on HPC.
  - Add more % for EDM generated data (there is more of it so we can go below 1%)
"""