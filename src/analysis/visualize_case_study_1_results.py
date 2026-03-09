"""
Produce components of Figure 5 (visualize the results of Case Study 1)
and parts of Figure 8 corresponding to Case Study 1.

Important information
---------------------
Ensure that `num_models_per_dataset` and `num-datasets` from config.py have the same values as they did during
running experiment2.py!
"""
import argparse

from src.config.config import get_config, ROOT
from src.data.loading import load_dataset
from src.models.loading import load_models_from_cs1
from src.utils.evaluation import compute_fairness_metrics, obtain_results, perform_paired_t_tests
from src.utils.io import load_results
from src.visualization.figures import plot_fairness
from src.visualization.tables import *


class PerformanceVisualizer:
    """Encapsulates all the necessary methods to perform the visualization pertaining to the results of
    experiment2.py"""
    def __init__(self, dataset_name: str):
        """Initialize the Visualizer class responsible for visualizing the improvement in fairness from using different
        hardness-based resampling techniques on the controlled pruning case study (experiment2.py).

        :param dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name

        config = get_config(args.dataset_name)
        self.num_classes = config['num_classes']
        self.num_epochs = config['num_epochs']
        self.num_training_samples = config['num_training_samples']
        self.num_test_samples = config['num_test_samples']
        self.num_models_per_dataset = config['num_models_per_dataset']
        self.num_datasets = config['num_datasets']

        self.figure_save_dir = os.path.join(ROOT, 'Figures/', args.dataset_name)
        self.results_dir = os.path.join(ROOT, "Results/")
        os.makedirs(self.figure_save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def main(self):
        """Main method for producing the visualizations."""
        models = load_models_from_cs1(self.dataset_name, self.num_epochs, self.num_datasets,
                                      self.num_models_per_dataset)
        _, _, test_loader, _ = load_dataset(self.dataset_name, apply_augmentation=True)

        results_save_dir = os.path.join(self.results_dir, self.dataset_name)
        file_name = "ensemble_results.pkl"
        results = obtain_results(results_save_dir, self.num_classes, test_loader, file_name, self.dataset_name, models)

        samples_per_class = load_results(os.path.join(self.results_dir, self.dataset_name, f'alpha_1',
                                                      'samples_per_class.pkl'))
        pruning_strategies = list(models.keys())
        fairness_results = compute_fairness_metrics(results, samples_per_class, pruning_strategies, self.num_classes)
        plot_fairness(fairness_results, self.figure_save_dir, 'pruning')
        fairness_t_test_results = perform_paired_t_tests(fairness_results, 'pruning')
        generate_t_test_table_for_fairness_metrics_for_pruning(fairness_t_test_results, self.figure_save_dir)
        generate_t_test_table_for_avg_values_for_pruning(fairness_t_test_results, self.figure_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., 'CIFAR10')")

    args = parser.parse_args()
    PerformanceVisualizer(args.dataset_name).main()
