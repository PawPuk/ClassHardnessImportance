"""
1. (+) Load the data
2. (+) Load hardness estimates (the model-based ones)
4. (+) Compute data-based hardness estimates (ID, Volume, Curvature, Class Impurity)
5. (+) Compute class-level estimates
6. (+) Compute Pearson and Spearman Rank correlation and create correlation matrix (add accuracies)
7. (+) Reduce the dimensionality of data using PCA and repeat the experiments with data-based hardness estimators
"""

import argparse
import collections
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perceptual_manifold_geometry import estimate_curvatures
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import torch
import torchvision
from tqdm import tqdm

from config import get_config, ROOT
from data import AugmentedSubset, load_dataset
from neural_networks import ResNet18LowRes
from utils import get_latest_model_index, load_hardness_estimates


class EstimatorBenchmarker:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        config = get_config(dataset_name)
        self.num_classes = config['num_classes']
        self.num_epochs = config['num_epochs']
        self.num_training_samples = sum(config['num_training_samples'])
        self.robust_num_models = config['robust_ensemble_size']
        self.model_dir = config['save_dir']
        self.save_epoch = config['save_epoch']

        self.results_save_dir = os.path.join(ROOT, 'Results/', f'unclean{dataset_name}')
        self.figures_save_dir = os.path.join(ROOT, 'Figures/', f'unclean{dataset_name}')
        for save_dir in [self.results_save_dir, self.figures_save_dir]:
            os.makedirs(save_dir, exist_ok=True)

        model_save_dir = os.path.join(config['save_dir'], 'none', f'unclean{dataset_name}')
        self.num_models = get_latest_model_index(model_save_dir, self.num_epochs) + 1

    def load_model(self, model_id: int) -> ResNet18LowRes:
        model = ResNet18LowRes(num_classes=self.num_classes).cuda()
        model_path = os.path.join(self.model_dir, 'none', f"unclean{args.dataset_name}",
                                  f'model_{model_id}_epoch_{self.save_epoch}.pth')

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            # This code can only be run if models were pretrained. If no pretrained models are found, throw error.
            raise Exception(f'Model {model_id} not found at epoch {self.save_epoch}.')

        return model

    def divide_samples_by_class(self, dataloader: torch.utils.data.DataLoader):
        all_inputs,  all_labels, all_indices = [], [], []
        samples_by_class = {c: [] for c in range(self.num_classes)}
        with torch.no_grad():
            for inputs, labels, indices in dataloader:
                inputs_cpu = inputs.cpu()  # [B, C, H, W]
                B = inputs_cpu.shape[0]
                flat_inputs = inputs_cpu.view(B, -1)  # [B, C*H*W]
                all_inputs.append(flat_inputs)
                all_labels.append(labels.cpu())
                all_indices.append(indices)
                # Group raw inputs by class
                for x, y in zip(flat_inputs, labels):
                    samples_by_class[y.item()].append(x.numpy())
        all_labels = torch.cat(all_labels, dim=0).numpy()
        return samples_by_class, all_labels

    def compute_class_impurity(self, samples_by_class, hardness_estimates, k: int = 15):
        all_samples, all_labels = [], []
        for class_label, class_samples in samples_by_class.items():
            all_samples.extend(class_samples)
            all_labels.extend([class_label] * len(class_samples))
        all_samples = np.stack(all_samples, axis=0)  # [N, D]
        all_labels = np.array(all_labels)  # [N]
        # Compute class impurity using k-NN
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(all_samples)
        distances, neighbors = nn.kneighbors(all_samples)  # shape: [N, k+1]
        impurity_by_class = collections.defaultdict(list)
        for idx in range(all_samples.shape[0]):
            sample_label = all_labels[idx]
            neighbor_indices = neighbors[idx][1:]  # Exclude self
            neighbor_labels = all_labels[neighbor_indices]
            impurity = np.sum(neighbor_labels != sample_label) / k
            impurity_by_class[sample_label].append(impurity)
        for c in self.num_classes:
            hardness_estimates['Class Impurity'][c] = float(np.mean(impurity_by_class[c]))

    @staticmethod
    def compute_intrinsic_dimension(samples_by_class, hardness_estimates):
        for class_label, class_samples in tqdm(samples_by_class.items(), desc='Iterating through classes.'):
            X = np.stack(class_samples, axis=0)  # [Nc, D]
            if X.shape[0] < 3:
                raise Exception  # Sanity check - this should not happen as no class should have only 2 samples.
            # Fit Nearest Neighbors
            nn = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X)
            distances, _ = nn.kneighbors(X)  # distances[i] = [self, T1, T2]
            T1 = distances[:, 1]  # First nearest neighbor
            T2 = distances[:, 2]  # Second nearest neighbor
            ratios = T2 / (T1 + 1e-8)
            # Remove any ratios <= 1 (log undefined or negative)
            valid_ratios = ratios[ratios > 1.0]
            if len(valid_ratios) == 0:
                raise Exception  # Sanity check - also shouldn't happen as this means we can't estimate class ID.
            else:
                log_mu = np.log(valid_ratios)
                d_hat = 1.0 / np.mean(log_mu)
                hardness_estimates['ID'][class_label] = d_hat

    @staticmethod
    def compute_volume(samples_by_class, hardness_estimates):
        for class_label, class_samples in tqdm(samples_by_class.items(), desc='Iterating through classes.'):
            if len(class_samples) < 2:
                raise Exception  # Sanity check - this should not happen as no class should have only 1 sample.
            X = np.stack(class_samples, axis=1)  # shape: [D, m]
            D, m = X.shape
            X_mean = np.mean(X, axis=1, keepdims=True)  # shape: [D, 1]
            X_centered = X - X_mean  # [D, m]
            S = (D / m) * (X_centered @ X_centered.T)  # [D, D]
            # Compute volume = 0.5 * log2(det(I + S))
            I = np.eye(D)
            try:
                det = np.linalg.det(I + S)
                if det <= 0:
                    raise Exception  # Sanity check - shouldn't happen as this means we can't estimate class volume.
                else:
                    volume = 0.5 * np.log2(det)
            except np.linalg.LinAlgError:
                raise Exception  # Sanity check - also shouldn't happen as this means we can't estimate class volume.
            hardness_estimates['Volume'][class_label] = volume

    @staticmethod
    def compute_curvature(samples_by_class, hardness_estimates, k: int = 15):
        for class_label, class_samples in tqdm(samples_by_class.items(), desc='Iterating through classes.'):
            intrinsic_dimension = hardness_estimates['ID'][class_label]
            curvature = estimate_curvatures(class_samples, k=k, pca_components=intrinsic_dimension,
                                            curvature_type='gaussian')
            hardness_estimates['Curvature'][class_label] = curvature

    def compute_data_based_hardness_estimates(self, hardness_estimates, samples_by_class):
        """
        Compute instance-level hardness scores using the final trained model.

        Each maps to a list of per-sample scores, ordered according to dataset indices.
        """
        for new_hardness_estimate in ['ID', 'Volume', 'Curvature']:
            hardness_estimates[new_hardness_estimate] = [0.0 for _ in range(self.num_classes)]
        hardness_estimates['Class Impurity'] = [0.0 for _ in range(self.num_training_samples)]
        t0 = time.time()
        print('Starting computing class impurities.')
        self.compute_class_impurity(samples_by_class, hardness_estimates)
        print(f'Finished computing class impurities in {time.time() - t0} seconds. Now computing intrinsic dimensions.')
        t0 = time.time()
        self.compute_intrinsic_dimension(samples_by_class, hardness_estimates)
        print(f'Finished computing intrinsic dimensions in {time.time() - t0} seconds. Now computing volumes.')
        t0 = time.time()
        self.compute_volume(samples_by_class, hardness_estimates)
        print(f'Finished computing volumes in {time.time() - t0} seconds. Now computing curvatures.')
        t0 = time.time()
        self.compute_curvature(samples_by_class, hardness_estimates)
        print(f'Finished computing volumes in {time.time() - t0} seconds.')

    def compute_class_level_accuracy(self, dataloader, hardness_estimates):
        print("Computing class-level accuracies...")
        t0 = time.time()
        class_correct = np.zeros((self.num_models, self.num_classes))
        class_total = np.zeros((self.num_models, self.num_classes))
        for model_id in tqdm(range(self.num_models), desc='iterating through models'):
            model = self.load_model(model_id)
            model.eval()
            with torch.no_grad():
                for inputs, labels, _ in dataloader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    for cls in range(self.num_classes):
                        class_mask = labels == cls
                        class_total[model_id, cls] += class_mask.sum().item()
                        class_correct[model_id, cls] += (preds[class_mask] == cls).sum().item()

        # Compute mean accuracy per class across models
        avg_accuracy_per_class = (class_correct / class_total).mean(axis=0)
        hardness_estimates['Accuracy'] = avg_accuracy_per_class.tolist()
        print(f"Finished computing class-level accuracies in {time.time() - t0} seconds.")

    def aggregate_model_based_estimates_to_class_level(self, hardness_estimates, labels):
        model_based_keys = [
            'Confidence', 'AUM', 'DataIQ', 'Forgetting', 'Loss',
            'iConfidence', 'iAUM', 'iDataIQ', 'iLoss', 'EL2N'
        ]
        for key in model_based_keys:
            estimates = np.array(hardness_estimates[key])  # shape [num_models][num_samples]
            sample_scores = np.mean(estimates, axis=0)  # [num_samples]
            class_sums = np.zeros(self.num_classes)
            class_counts = np.zeros(self.num_classes)
            for score, label in zip(sample_scores, labels):
                class_sums[label] += score
                class_counts[label] += 1
            class_averages = class_sums / np.maximum(class_counts, 1e-8)
            hardness_estimates[key] = class_averages

    def compute_and_plot_correlations(self, hardness_estimates, projected: str = ''):
        keys = list(hardness_estimates.keys())
        matrix = np.array([hardness_estimates[k] for k in keys])  # [num_metrics, num_classes]
        df = pd.DataFrame(matrix.T, columns=keys)

        # Plot Pearson correlation matrix
        pearson_corr = df.corr(method='pearson')
        plt.figure(figsize=(10, 8))
        sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f"Pearson Correlation Between {projected.capitalize()} Class-Level Hardness Estimators")
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_save_dir, f"{projected}_correlation_matrix_pearson.pdf"))
        plt.close()

        # Plot Spearman correlation matrix
        spearman_corr = df.corr(method='spearman')
        plt.figure(figsize=(10, 8))
        sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f"Spearman Correlation Between {projected.capitalize()} Hardness Estimators")
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_save_dir, f"{projected}_correlation_matrix_spearman.pdf"))
        plt.close()

    @staticmethod
    def project_classes_via_pca(samples_by_class, hardness_estimates):
        print("Projecting each class via PCA using 2 × estimated ID.")
        projected_samples_by_class = {}
        for class_label, class_samples in tqdm(samples_by_class.items()):
            X = np.stack(class_samples, axis=0)  # [Nc, D]
            intrinsic_dim = hardness_estimates['ID'][class_label]
            reduced_dim = min(int(intrinsic_dim * 2), X.shape[1])  # Make sure it doesn’t exceed original dim

            # Fit PCA on centered class samples
            pca = PCA(n_components=reduced_dim)
            X_reduced = pca.fit_transform(X)
            projected_samples_by_class[class_label] = [x for x in X_reduced]

        return projected_samples_by_class

    def main(self):
        config = get_config(self.dataset_name)
        _, training_dataset, _, _ = load_dataset(args.dataset_name, 'unclean', False, False)
        new_training_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(config['mean'], config['std']),
        ])
        training_dataset = AugmentedSubset(training_dataset, transform=new_training_transform)
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1000, shuffle=False)
        samples_by_class, labels = self.divide_samples_by_class(training_loader)
        hardness_estimates = load_hardness_estimates('unclean', self.dataset_name)
        self.compute_class_level_accuracy(training_loader, hardness_estimates)
        self.aggregate_model_based_estimates_to_class_level(hardness_estimates, labels)
        self.compute_data_based_hardness_estimates(hardness_estimates, samples_by_class)
        self.compute_and_plot_correlations(hardness_estimates)
        projected_samples_by_class = self.project_classes_via_pca(samples_by_class, hardness_estimates)
        print('Now repeating the experiments on data projected onto low dimensional representations obtained via PCA.')
        self.compute_data_based_hardness_estimates(hardness_estimates, projected_samples_by_class)
        self.compute_and_plot_correlations(hardness_estimates, 'projected')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Hardness of Dataset Samples')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='Specify the dataset name (default: CIFAR10)')
    args = parser.parse_args()

    EstimatorBenchmarker(args.dataset_name).main()
