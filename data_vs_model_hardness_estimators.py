"""
1. (+) Load the data
2. (+) Load hardness estimates (the model-based ones)
4. (+) Compute data-based hardness estimates (ID, Volume, Curvature, Class Impurity)
5. (+) Compute class-level estimates
6. (+) Compute Pearson and Spearman Rank correlation and create correlation matrix (add accuracies)
7. (+) Reduce the dimensionality of data using PCA and repeat the experiments with data-based hardness estimators
8. (+) Use both subjective (my ensemble) and objective (ensemble of different architectures from GitHub) for accuracy
9. (+) Plot the class-level ID estimates (sorted and unsorted)
10. (+) Plot ID as a function of the dimensionality of the space (use PCA projections)
11. (+) Modify the projection experiments to iterate through dimensions rather than use only one, hardcoded dimension.
12. (+) Add correlations between volume and ID as a function of projection dimension.
"""

import argparse
import collections
import os
import pickle
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
        self.num_training_samples = config['num_training_samples']
        self.robust_num_models = config['robust_ensemble_size']
        self.model_dir = config['save_dir']
        self.save_epoch = config['save_epoch']

        self.results_save_dir = os.path.join(ROOT, 'Results/', f'unclean{dataset_name}')
        self.figures_save_dir = os.path.join(ROOT, 'Figures/', f'unclean{dataset_name}')
        for save_dir in [self.results_save_dir, self.figures_save_dir]:
            os.makedirs(save_dir, exist_ok=True)

        model_save_dir = os.path.join(config['save_dir'], 'none', f'unclean{dataset_name}')
        self.num_models = get_latest_model_index(model_save_dir, self.num_epochs) + 1

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

    def load_objective_models(self):
        dataset_name = 'cifar10' if self.dataset_name == 'CIFAR10' else 'cifar100'
        supported_models = (
            f"{dataset_name}_resnet20", f"{dataset_name}_resnet32", f"{dataset_name}_resnet44",
            f"{dataset_name}_resnet56", f"{dataset_name}_vgg11_bn", f"{dataset_name}_vgg13_bn",
            f"{dataset_name}_vgg16_bn", f"{dataset_name}_vgg19_bn", f"{dataset_name}_mobilenetv2_x0_5",
            f"{dataset_name}_mobilenetv2_x0_75", f"{dataset_name}_mobilenetv2_x1_0", f"{dataset_name}_mobilenetv2_x1_4",
            f"{dataset_name}_shufflenetv2_x0_5", f"{dataset_name}_shufflenetv2_x1_0",
            f"{dataset_name}_shufflenetv2_x1_5", f"{dataset_name}_shufflenetv2_x2_0",
            f"{dataset_name}_shufflenetv2_x0_5", f"{dataset_name}_repvgg_a0", f"{dataset_name}_repvgg_a1",
            f"{dataset_name}_repvgg_a2"
        )

        models = []
        for model_name in supported_models:
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True)
            model = model.eval().cuda()
            models.append(model)

        return models

    def compute_class_level_accuracy(self, dataloader, hardness_estimates):
        print("Computing class-level accuracies...")
        t0 = time.time()

        def evaluate_ensemble(models, label):
            class_correct = np.zeros((self.num_models, self.num_classes))
            class_total = np.zeros((self.num_models, self.num_classes))
            for model_id, model in tqdm(enumerate(models), desc='iterating through models'):
                with torch.no_grad():
                    for inputs, labels, _ in dataloader:
                        inputs, labels = inputs.cuda(), labels.cuda()
                        outputs = model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        for cls in range(self.num_classes):
                            class_mask = labels == cls
                            class_total[model_id, cls] += class_mask.sum().item()
                            class_correct[model_id, cls] += (preds[class_mask] == cls).sum().item()
            avg_accuracy_per_class = (class_correct / class_total).mean(axis=0)
            hardness_estimates[label] = avg_accuracy_per_class.tolist()

        subjective_models = [self.load_model(model_id).eval().cuda() for model_id in range(self.num_models)]
        evaluate_ensemble(subjective_models, label='Subjective Accuracy')
        objective_models = self.load_objective_models()
        evaluate_ensemble(objective_models, label='Objective Accuracy')
        print(f"Finished computing class-level accuracies in {time.time() - t0} seconds.")

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
        for c in range(self.num_classes):
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
                hardness_estimates['ID'][class_label] = int(round(d_hat))

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
                    eigvals = np.linalg.eigvalsh(S)
                    volume = 0.5 * np.sum(np.log2(1 + eigvals))
            except np.linalg.LinAlgError:
                raise Exception  # Sanity check - also shouldn't happen as this means we can't estimate class volume.
            hardness_estimates['Volume'][class_label] = volume

    @staticmethod
    def compute_curvature(samples_by_class, hardness_estimates, d):
        for class_label, class_samples in tqdm(samples_by_class.items(), desc='Iterating through classes.'):
            intrinsic_dimension = hardness_estimates['ID'][class_label]
            class_samples = np.stack(class_samples, axis=0)  # Ensure it's a 2D NumPy array: [Nc, D]
            d = intrinsic_dimension if d is None else d
            curvature = estimate_curvatures(class_samples, k=2*d, pca_components=d, curvature_type='gaussian')
            hardness_estimates['Curvature'][class_label] = curvature

    def compute_data_based_hardness_estimates(self, hardness_estimates, samples_by_class,
                                              projected_dimensionality=None, projected_samples_by_class=None):
        """
        Compute instance-level hardness scores using the final trained model.

        Each maps to a list of per-sample scores, ordered according to dataset indices.
        """
        for new_hardness_estimate in ['ID', 'Volume', 'Curvature']:
            hardness_estimates[new_hardness_estimate] = [0.0 for _ in range(self.num_classes)]
        if projected_dimensionality is None:
            hardness_estimates['Class Impurity'] = [0.0 for _ in range(sum(self.num_training_samples))]
            self.compute_class_impurity(samples_by_class, hardness_estimates)
            projected_samples_by_class = samples_by_class
        self.compute_intrinsic_dimension(projected_samples_by_class, hardness_estimates)
        self.compute_volume(projected_samples_by_class, hardness_estimates)
        if projected_dimensionality is None or projected_dimensionality < 35:  # To reduce computational cost
            self.compute_curvature(samples_by_class, hardness_estimates, projected_dimensionality)

    def aggregate_model_based_estimates_to_class_level(self, hardness_estimates, labels):
        estimates_to_aggregate = [
            'Confidence', 'AUM', 'DataIQ', 'Forgetting', 'Loss',
            'iConfidence', 'iAUM', 'iDataIQ', 'iLoss', 'EL2N', 'Class Impurity'
        ]
        for key in estimates_to_aggregate:
            sample_scores = np.array(hardness_estimates[key])  # shape [num_models][num_samples]
            if key != 'Class Impurity':
                sample_scores = np.mean(sample_scores, axis=0)  # [num_samples]
            class_sums = np.zeros(self.num_classes)
            class_counts = np.zeros(self.num_classes)
            for score, label in zip(sample_scores, labels):
                class_sums[label] += score
                class_counts[label] += 1
            class_averages = class_sums / np.maximum(class_counts, 1e-8)
            hardness_estimates[key] = class_averages

    def plot_class_level_id_estimates(self, hardness_estimates):
        """Plot class-level intrinsic dimension (ID) estimates: sorted and unsorted."""
        id_estimates = np.array(hardness_estimates['ID'])
        sorted_id = np.sort(id_estimates)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(len(sorted_id)), sorted_id, marker='o', linestyle='-', color='salmon')
        ax.set_xlabel("Sorted Class Index")
        ax.set_ylabel("Intrinsic Dimension")
        ax.set_title("Class-Level Intrinsic Dimensions (Sorted)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.yaxis.get_major_locator().set_params(integer=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_save_dir, "class_id_sorted.pdf"))
        plt.close()

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
        plt.savefig(os.path.join(self.figures_save_dir, f"{projected}correlation_matrix_pearson.pdf"))
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
    def project_classes_via_pca(samples_by_class, reduced_dim):
        projected_samples_by_class = {}
        for i, (class_label, class_samples) in tqdm(enumerate(samples_by_class.items())):
            X = np.stack(class_samples, axis=0)  # [Nc, D]
            # Fit PCA on centered class samples
            pca = PCA(n_components=reduced_dim)
            X_reduced = pca.fit_transform(X)
            projected_samples_by_class[class_label] = [x for x in X_reduced]
        return projected_samples_by_class

    def load_or_compute_projected_estimates(self, samples_by_class, hardness_estimates):
        cache_path = os.path.join(self.results_save_dir, 'projected_estimates.pkl')
        if os.path.exists(cache_path):
            print(f"Loading projected estimates from {cache_path}")
            with open(cache_path, 'rb') as f:
                projected_estimates = pickle.load(f)
            return projected_estimates

        dimensionalities = [10, 12, 15, 17, 20, 22, 25, 30, 35, 40, 45, 50, 75, 100]
        print("Computing projected estimates...")
        projected_estimates = {
            metric_name: {
                dimensionality: [] for dimensionality in dimensionalities
            } for metric_name in ['ID', 'Volume', 'Curvature']
        }

        for dimensionality in tqdm(dimensionalities, desc='Iterating through dimensionalities'):
            projected_samples_by_class = self.project_classes_via_pca(samples_by_class, dimensionality)
            self.compute_data_based_hardness_estimates(hardness_estimates, samples_by_class, dimensionality,
                                                       projected_samples_by_class)
            for metric_name in projected_estimates:
                projected_estimates[metric_name][dimensionality] = hardness_estimates[metric_name]

        with open(cache_path, 'wb') as f:
            pickle.dump(projected_estimates, f)
            print(f"Saved projected estimates to {cache_path}")
        return projected_estimates

    def plot_projection_correlation_trends(self, projected_estimates, hardness_estimates):
        dimensionalities = sorted(next(iter(projected_estimates.values())).keys(), reverse=True)
        metric_names = list(projected_estimates.keys())
        obj = np.array(hardness_estimates['Objective Accuracy'])
        subj = np.array(hardness_estimates['Subjective Accuracy'])

        for metric in metric_names:
            pearsons, spearmans = [], []
            for dimensionality in dimensionalities:
                metric_vals = np.array(projected_estimates[metric][dimensionality])
                pearson = np.corrcoef(metric_vals, obj)[0, 1], np.corrcoef(metric_vals, subj)[0, 1]
                spearman = pd.Series(metric_vals).corr(pd.Series(obj), method='spearman'), \
                    pd.Series(metric_vals).corr(pd.Series(subj), method='spearman')
                pearsons.append(pearson)
                spearmans.append(spearman)

            pearsons = np.array(pearsons)
            spearmans = np.array(spearmans)

            plt.figure(figsize=(8, 6))
            plt.plot(dimensionalities, pearsons[:, 0], label='Pearson (Objective)', color='blue')
            plt.plot(dimensionalities, pearsons[:, 1], label='Pearson (Subjective)', color='blue', linestyle='--')
            plt.plot(dimensionalities, spearmans[:, 0], label='Spearman (Objective)', color='green')
            plt.plot(dimensionalities, spearmans[:, 1], label='Spearman (Subjective)', color='green', linestyle='--')
            plt.xlabel('Dimensionality')
            plt.ylabel('Correlation')
            plt.title(f'Correlation of {metric} vs Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.figures_save_dir, f"{metric}_projection_correlations.pdf"))
            plt.close()

    def plot_inter_metric_correlation_trends(self, projected_estimates):
        metric_pairs = [('ID', 'Volume')]
        dimensionalities = sorted(next(iter(projected_estimates.values())).keys())

        for metric_x, metric_y in metric_pairs:
            pearsons, spearmans = [], []
            for dimensionality in dimensionalities:
                x = np.array(projected_estimates[metric_x][dimensionality])
                y = np.array(projected_estimates[metric_y][dimensionality])
                pearsons.append(np.corrcoef(x, y)[0, 1])
                spearmans.append(pd.Series(x).corr(pd.Series(y), method='spearman'))

            plt.figure(figsize=(8, 6))
            plt.plot(dimensionalities, pearsons, label='Pearson', color='blue')
            plt.plot(dimensionalities, spearmans, label='Spearman', color='green')
            plt.xlabel('PCA Dimensionality')
            plt.ylabel('Correlation')
            plt.title(f'{metric_x} vs {metric_y} Correlation by PCA Dimensionality')
            plt.legend()
            plt.grid(True)
            fname = f"{metric_x}_vs_{metric_y}_projection_correlation.pdf"
            plt.savefig(os.path.join(self.figures_save_dir, fname))
            plt.close()

    def main(self):
        config = get_config(self.dataset_name)
        _, training_dataset, _, _ = load_dataset(args.dataset_name, False, False, False)
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
        self.compute_data_based_hardness_estimates(hardness_estimates, samples_by_class)
        self.aggregate_model_based_estimates_to_class_level(hardness_estimates, labels)
        if 'probs' in hardness_estimates.keys():
            del hardness_estimates['probs']
        self.plot_class_level_id_estimates(hardness_estimates)
        self.compute_and_plot_correlations(hardness_estimates)

        del hardness_estimates['Class Impurity']
        projected_estimates = self.load_or_compute_projected_estimates(samples_by_class, hardness_estimates)
        self.plot_projection_correlation_trends(projected_estimates, hardness_estimates)
        self.plot_inter_metric_correlation_trends(projected_estimates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Hardness of Dataset Samples')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='Specify the dataset name (default: CIFAR10)')
    args = parser.parse_args()

    EstimatorBenchmarker(args.dataset_name).main()
