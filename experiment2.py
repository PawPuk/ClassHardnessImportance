"""This is the second important module that allows training the ensembles on pruned subdatasets."""

import argparse
import os.path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader

from data_pruning import DataPruning
from train_ensemble import ModelTrainer
from config import get_config, ROOT
from data import AugmentedSubset, IndexedDataset, load_dataset, perform_data_augmentation
from utils import compute_sample_allocation_after_resampling, load_hardness_estimates, set_reproducibility


class Experiment2:
    """Encapsulates all the necessary methods to perform the pruning experiments."""
    def __init__(self, pruning_strategy: str, dataset_name: str, pruning_rate: int, hardness_estimator: str,
                 oversampling_strategy: str):
        """Initialize the Experiment2 class with configuration specific to the current experiment.

        :param pruning_strategy: Specifies the pruning strategy. The only viable options are `clp` and `dlp`. The former
        indicates that pruning will be performed at class level ensuring balanced pruned subdatasets. The latter
        indicates dataset-level pruning and requires `oversampling_strategy` to be specified. Ths is because dlp
        operates by first performing clp and then applying hardness-based resampling on the pruned subdataset.
        :param dataset_name: Name of the dataset.
        :param pruning_rate: An integer specifying the percentage of samples that will be pruned from the dataset.
        :param hardness_estimator: Name of the hardness estimator that will be used to compute the resampling ratios,
        which specifies how many samples to keep in each class after hardness-based resampling.
        :param oversampling_strategy: Name of the oversampling strategy for dlp.
        """
        set_reproducibility()

        self.dataset_name = dataset_name
        self.pruning_strategy = pruning_strategy
        self.pruning_rate = pruning_rate
        self.hardness_estimator = hardness_estimator
        self.oversampling_strategy = oversampling_strategy

        # Constants taken from config
        config = get_config(dataset_name)
        self.BATCH_SIZE = config['batch_size']
        self.SAVE_EPOCH = config['save_epoch']
        self.MODEL_DIR = config['save_dir']
        self.NUM_CLASSES = config['num_classes']
        self.NUM_EPOCHS = config['num_epochs']
        self.NUM_TRAINING_SAMPLES = sum(config['num_training_samples'])
        self.DATASET_COUNT = config['num_datasets']
        self.NUM_MODELS_FOR_HARDNESS = config['num_models_for_hardness']

        self.figure_save_dir = os.path.join(ROOT, 'Figures/')

    def investigate_resampling_ratios(self, hardness_estimates: Dict[Tuple[int, int], Dict[str, List[float]]],
                                      labels: List[int], thresholds: List[int]
                                      ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[int]]]:
        """The resampling ratios can be obtained through various means. Currently, we are computing them using the
        formulations from compute_imbalance_ratio_with_hardness() method, but they can be also obtained through simple
        pruning. If we perform dataset-level pruning focusing on removing easy sample we inadvertently produce
        imbalanced pruned subdatasets. This imbalance could be also used as resampling ratio. The idea behind this
        method was to check if the resampling ratios computed through this pruning approach are stable across pruning
        thresholds - if they are (and they are) then it gives them legitimacy. However, pursuing this direction requires
        more work and is related to modifying the hardness-based resampling by focusing on the algorithm that produces
        resampling_ratios. This is interesting future direction but beyond our current scope.
        """
        per_class_counts, pearson_scores, spearman_scores, ideal_ratios, window_size = {}, {}, {}, {}, 5
        for estimator_name in hardness_estimates[(0, 0)].keys():
            if estimator_name == 'probs':
                continue
            hardness_over_models = [hardness_estimates[(0, model_id)][estimator_name]
                                    for model_id in range(len(hardness_estimates))]
            estimates = np.mean(np.array(hardness_over_models[:self.NUM_MODELS_FOR_HARDNESS]), axis=0)
            sorted_indices = np.argsort(estimates)
            num_samples = len(estimates)
            per_class_counts[estimator_name] = []
            pearson_scores[estimator_name], spearman_scores[estimator_name] = [], []
            for t in thresholds:
                retain_count = int((100 - t) / 100 * num_samples)
                retained_indices = sorted_indices[:retain_count]
                counts = np.zeros(self.NUM_CLASSES, dtype=int)
                for idx in retained_indices:
                    cls = labels[idx]
                    counts[cls] += 1
                per_class_counts[estimator_name].append(counts)
            for i in range(len(per_class_counts[estimator_name]) - 1):
                x = per_class_counts[estimator_name][i]
                y = per_class_counts[estimator_name][i + 1]
                pearson_scores[estimator_name].append(np.corrcoef(x, y)[0, 1])
                spearman_scores[estimator_name].append(pd.Series(x).corr(pd.Series(y), method='spearman'))
            pearson_avg = pd.Series(pearson_scores[estimator_name]).rolling(window=window_size, min_periods=1).mean()
            spearman_avg = pd.Series(spearman_scores[estimator_name]).rolling(window=window_size, min_periods=1).mean()
            combined_avg = (pearson_avg + spearman_avg) / 2
            best_combined_idx = combined_avg.idxmax()
            ideal_ratios[estimator_name] = per_class_counts[estimator_name][best_combined_idx]
            print(f'{best_combined_idx} is the best ratio for {estimator_name}.')
        hardness_over_models = [hardness_estimates[(0, model_id)][self.hardness_estimator]
                                for model_id in range(len(hardness_estimates))]
        estimates = list(np.mean(np.array(hardness_over_models[:self.NUM_MODELS_FOR_HARDNESS]), axis=0))
        # Hardness-Based Resampling Ratio computation
        ideal_ratios['HBRR'], _ = compute_sample_allocation_after_resampling(estimates, labels, self.NUM_CLASSES,
                                                                             self.NUM_TRAINING_SAMPLES,
                                                                             self.hardness_estimator, self.pruning_rate)
        return pearson_scores, spearman_scores, ideal_ratios

    def measure_stability_of_resampling_ratios(self, pearson_scores: Dict[str, List[float]],
                                               spearman_scores: Dict[str, List[float]],
                                               ideal_ratios: Dict[str, List[int]],
                                               hardness_estimates:  Dict[Tuple[int, int], Dict[str, List[float]]],
                                               thresholds: List[int]):
        """Produces the visualizations for the stability experiments performed in investigate_resampling_ratios()."""
        for estimator_name in hardness_estimates[(0, 0)].keys():
            if estimator_name == 'probs':
                continue
            plt.figure(figsize=(8, 5))
            plt.plot(thresholds[:-1], pearson_scores[estimator_name], label='Pearson', color='blue')
            plt.plot(thresholds[:-1], spearman_scores[estimator_name], label='Spearman', color='orange')
            plt.xlabel("Pruning Threshold (%)")
            plt.ylabel("Inter-Class Distribution Correlation")
            plt.title(f"Class Retention Correlation Across Pruning Thresholds ({estimator_name})")
            plt.legend()
            plt.grid(True)
            fig_path = os.path.join(self.figure_save_dir, f"unclean{self.dataset_name}",
                                    f"correlation_threshold_stability_{estimator_name}.pdf")
            plt.savefig(fig_path)
            plt.close()

        ideal_ratios_df = pd.DataFrame(ideal_ratios)
        correlation_matrix = ideal_ratios_df.corr()
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Pearson Correlation Matrix of Ideal Resampling Ratios")
        fig.savefig(os.path.join(self.figure_save_dir, f"unclean{self.dataset_name}",
                                 "resampling_ratio_pearson_correlation_matrix.pdf"))
        plt.close()
        correlation_matrix = ideal_ratios_df.corr(method='spearman')
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Spearman Correlation Matrix of Ideal Resampling Ratios")
        fig.savefig(os.path.join(self.figure_save_dir, f"unclean{self.dataset_name}",
                                 "resampling_ratio_spearman_correlation_matrix.pdf"))
        plt.close()

    def prune_dataset(self, labels: List[int], training_dataset: Union[AugmentedSubset, IndexedDataset],
                      imbalance_ratio: List[int]) -> AugmentedSubset:
        """Produce the pruned dataset through DataPruning"""
        pruner = DataPruning(self.pruning_rate, self.dataset_name, imbalance_ratio)
        if self.pruning_strategy == 'dlp':
            pruned_subdataset = pruner.resampling_pruned_subdataset(self.oversampling_strategy, labels,
                                                                    training_dataset)
        elif self.pruning_strategy == 'clp':
            pruned_subdataset, _ = pruner.class_level_pruning(labels, training_dataset)
        else:
            raise ValueError('Wrong value of the parameter `pruning_strategy`.')
        augmented_subdataset = perform_data_augmentation(pruned_subdataset, self.dataset_name)
        return augmented_subdataset

    def run_experiment(self):
        """Main method for running the experiments."""
        _, training_dataset, test_loader, _ = load_dataset(self.dataset_name)
        labels = [training_dataset[idx][1].item() for idx in range(len(training_dataset))]

        hardness_estimates = load_hardness_estimates('unclean', self.dataset_name)
        thresholds = [i for i in range(1, 100, 1)]
        pearson_scores, spearman_scores, ideal_ratios = self.investigate_resampling_ratios(hardness_estimates, labels,
                                                                                           thresholds)
        self.measure_stability_of_resampling_ratios(pearson_scores, spearman_scores, ideal_ratios, hardness_estimates,
                                                    thresholds)
        pruned_training_loaders = []
        for _ in range(self.DATASET_COUNT):
            pruned_dataset = self.prune_dataset(labels, training_dataset, ideal_ratios['HBRR'])
            pruned_training_loaders.append(DataLoader(pruned_dataset, batch_size=self.BATCH_SIZE, shuffle=True,
                                                      num_workers=2))

        trainer = ModelTrainer(len(training_dataset), pruned_training_loaders, test_loader, self.dataset_name,
                               f"{self.oversampling_strategy}{self.pruning_strategy}{self.pruning_rate}",
                               False)
        trainer.train_ensemble()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EL2N Score Calculation and Dataset Pruning')
    parser.add_argument('--pruning_strategy', type=str, choices=['clp', 'dlp'],
                        help='Choose pruning strategy: clp (fixed class level pruning) or dlp (data level pruning)')
    parser.add_argument('--dataset_name', type=str, choices=['CIFAR10', 'CIFAR100'],
                        help='Specify the dataset name (default: CIFAR10)')
    parser.add_argument('--pruning_rate', type=int,
                        help='Percentage of data samples that will be removed during data pruning (use integers).')
    parser.add_argument('--hardness_estimator', type=str, default='AUM',
                        help='Specifies which hardness estimator to use for pruning.')
    parser.add_argument('--oversampling_strategy', type=str, choices=['random', 'SMOTE', 'holdout'],
                        help='Specifies what oversampling to use (only applicable for dlp)')

    args = parser.parse_args()

    # Initialize and run the experiment
    experiment = Experiment2(args.pruning_strategy, args.dataset_name, args.pruning_rate, args.hardness_estimator,
                             args.oversampling_strategy)
    experiment.run_experiment()


# TODO: Implement holdout oversampling
