"""
Produce components of Figure 6 (visualize the results of Case Study 2)
and parts of Figure 8 corresponding to Case Study 2.


Main purpose
-------------------
* Compare Recall and Precision between ensembles trained on: (i) original datasets; and (ii) dataset variants obtained through hardness-based resampling.
* Measure and visualize the changes to fairness coming from hardness-based resampling using gap- and dispersion-based metrics.

Important information
-------------------
* Ensure that `num_models_per_dataset` and `num_datasets` from config.py have the same values as they did during
running experiment3.py!
"""
import argparse

from src.config.config import get_config, ROOT
from src.data.loading import load_dataset
from src.models.loading import load_models_from_cs2
from src.utils.evaluation import compute_fairness_metrics, obtain_results, perform_paired_t_tests
from src.utils.io import load_sample_allocations
from src.visualization.tables import *
from src.visualization.figures import plot_fairness, visualize_resampling_results


class ResamplingVisualizer:
    """Encapsulates all the necessary methods to perform the visualization pertaining to the results of
    experiments3.py."""
    def __init__(self, dataset_name):
        """Initialize the Visualizer class responsible for visualizing the improvement in fairness from using different
         hardness-based resampling techniques on the full data (experiment3.py).

         :param dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name

        config = get_config(args.dataset_name)
        self.num_classes = config['num_classes']
        self.num_training_samples = config['num_training_samples']
        self.num_test_samples = config['num_test_samples']
        self.num_epochs = config['num_epochs']
        self.num_models_per_dataset = config['num_models_per_dataset']
        self.num_datasets = config['num_datasets']

        self.figure_save_dir = os.path.join(ROOT, 'Figures/', dataset_name)
        self.hardness_save_dir = os.path.join(ROOT, f"Results/")

        for save_dir in self.figure_save_dir, self.hardness_save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def main(self):
        """Main method for producing the visualizations."""
        results_dir = os.path.join(ROOT, "Results", self.dataset_name)
        models = load_models_from_cs2(self.dataset_name, self.num_epochs, self.num_datasets,
                                      self.num_models_per_dataset)
        _, _, test_loader, _ = load_dataset(self.dataset_name, apply_augmentation=True)

        results = obtain_results(results_dir, self.num_classes, test_loader, "resampling_results.pkl",
                                 self.dataset_name, models)

        samples_per_class = load_sample_allocations(self.hardness_save_dir, self.dataset_name)
        _, _ = visualize_resampling_results(samples_per_class, self.num_classes, self.dataset_name,
                                            self.figure_save_dir)

        resampling_strategies = list(results['Recall'].keys())
        fairness_results = compute_fairness_metrics(results, samples_per_class[1], resampling_strategies,
                                                    self.num_classes)

        plot_fairness(fairness_results, self.figure_save_dir, 'resampling')
        t_test_results = perform_paired_t_tests(fairness_results, 'resampling')
        generate_t_test_table_for_fairness_metrics_for_resampling(t_test_results, self.figure_save_dir)
        generate_t_test_table_for_avg_values_for_resampling(t_test_results, self.figure_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (e.g., 'CIFAR10')")

    args = parser.parse_args()
    ResamplingVisualizer(args.dataset_name).main()
