"""This is the second visualization module that focuses on the results of experiment2.py

Main purpose:
*

Important information:
* Ensure that `num_models_per_dataset` and `num-datasets` from config.py have the same values as they did during
running experiment2.py!

"""
import argparse

from fiona.crs import defaultdict

from data import load_dataset
from config import get_config
from utils import *


class PerformanceVisualizer:
    """Encapsulates all the necessary methods to perform the visualization pertaining to the results of
    experiment2.py"""
    def __init__(self, dataset_name: str, alpha: str):
        """Initialize the Visualizer class responsible for visualizing the improvement in fairness from using different
        hardness-based resampling techniques on the controlled pruning case study (experiment2.py).

        :param dataset_name: Name of the dataset
        :param alpha: The module will produce visualization for this particular value of alpha (assuming that the
        experiments were run on this setting).
        """
        self.dataset_name = dataset_name
        self.alpha = int(alpha)

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

    def load_models(self) -> Dict[str, Dict[int, Dict[int, List[str]]]]:
        """Used to load pretrained models."""
        models_dir = os.path.join(ROOT, "Models/")
        models_by_strategy, pruning_rate = defaultdict(lambda: defaultdict(lambda: defaultdict(list))), None

        for pruning_strategy in ['none', 'random', 'holdout', 'SMOTE']:
            # Walk through each folder in the Models directory
            for root, dirs, files in os.walk(models_dir):
                # Ensure the dataset name matches exactly (avoid partial matches like "cifar10" in "cifar100")
                if f"{pruning_strategy}_pruning" in root and os.path.basename(root) == f'unclean{self.dataset_name}':
                    pruning_rate = int(root.split("pruning_rate_")[1].split("_alpha_")[0])
                    alpha = int(root.split("_alpha_")[1].split("/")[0])

                    for file in files:
                        if file.endswith(".pth") and f"_epoch_{self.num_epochs}" in file and alpha == self.alpha:
                            dataset_index = int(file.split("_")[1])
                            model_index = int(file.split("_")[3])
                            if dataset_index >= self.num_datasets or model_index >= self.num_models_per_dataset:
                                raise Exception('The `num_datasets` and `num_models_per_dataset` in config.py needs to '
                                                'have the same values as it when running experiment3.py.')

                            model_path = os.path.join(root, file)
                            models_by_strategy[pruning_strategy][pruning_rate][dataset_index].append(model_path)
                    if len(models_by_strategy[pruning_strategy][pruning_rate]) > 0:
                        for i in range(1, len(models_by_strategy[pruning_strategy][pruning_rate])):  # Sanity check
                            assert len(models_by_strategy[pruning_strategy][pruning_rate][0]) == \
                                   len(models_by_strategy[pruning_strategy][pruning_rate][i])
                        print(
                            f"Loaded {len(models_by_strategy[pruning_strategy][pruning_rate])} ensembles of models for "
                            f"strategies {pruning_strategy} and pruning rate {pruning_rate}, with each ensemble having "
                            f"{len(models_by_strategy[pruning_strategy][pruning_rate][0])} models.")

            print(f"Models loaded by pruning rate for {pruning_strategy} on {self.dataset_name}")

        print(models_by_strategy.keys())
        print(models_by_strategy)
        return defaultdict_to_dict(models_by_strategy)

    def main(self):
        """Main method for producing the visualizations."""
        models = self.load_models()
        _, _, test_loader, _ = load_dataset(self.dataset_name, apply_augmentation=True)

        results = obtain_results(os.path.join(self.results_dir, self.dataset_name), self.num_classes, test_loader,
                                 "ensemble_results.pkl", models)

        samples_per_class = load_results(os.path.join(self.results_dir, f'unclean{self.dataset_name}', f'alpha_1',
                                                      'samples_per_class.pkl'))
        pruning_strategies = list(models.keys())
        fairness_results = compute_fairness_metrics(results, samples_per_class, pruning_strategies, self.num_classes)
        plot_fairness_dual_axis(fairness_results, self.figure_save_dir, 'pruning')
        fairness_t_test_results = perform_paired_t_tests(fairness_results, 'pruning')
        generate_t_test_table_for_fairness_metrics_for_pruning(fairness_t_test_results, self.figure_save_dir)
        generate_t_test_table_for_avg_values_for_pruning(fairness_t_test_results, self.figure_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load models for specified pruning strategy and dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., 'CIFAR10')")
    parser.add_argument('--alpha', type=int, default=1, help='Value of the alpha parameter for which we want to produce'
                                                             ' visualizations (see experiment2.py for details).')
    args = parser.parse_args()
    PerformanceVisualizer(args.dataset_name, args.alpha).main()
