import pickle
import os
import argparse


class Experiment3:
    def __init__(self, dataset_name, desired_dataset_size):
        self.dataset_name = dataset_name
        self.desired_dataset_size = desired_dataset_size
        self.results_file = os.path.join('Results', dataset_name, 'el2n_scores.pkl')

    def load_results(self):
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        with open(self.results_file, 'rb') as file:
            _, _, _, self.dataset_accuracies, self.class_accuracies = pickle.load(file)

    def compute_average_accuracies(self):
        # Calculate the average dataset-level accuracy
        avg_dataset_accuracy = sum(self.dataset_accuracies) / len(self.dataset_accuracies)

        # Aggregate class-level accuracies
        class_accumulator = {i: 0 for i in range(len(self.class_accuracies[0]))}
        class_counts = {i: 0 for i in range(len(self.class_accuracies[0]))}

        for model_class_acc in self.class_accuracies:
            for class_id, acc in model_class_acc.items():
                class_accumulator[class_id] += acc
                class_counts[class_id] += 1

        avg_class_accuracies = {class_id: class_accumulator[class_id] / class_counts[class_id]
                                for class_id in class_accumulator}

        # Return results
        return avg_dataset_accuracy, avg_class_accuracies, len(self.dataset_accuracies)

    @staticmethod
    def print_results(avg_dataset_accuracy, avg_class_accuracies, num_models):
        print(f"Number of Models: {num_models}")
        print(f"Average Dataset-Level Accuracy: {avg_dataset_accuracy:.2%}")
        print("Average Class-Level Accuracies:")
        for class_id, acc in avg_class_accuracies.items():
            print(f"  Class {class_id}: {acc:.2%}")

    @staticmethod
    def compute_hardness_based_ratios(avg_class_accuracies):
        # Compute hardness-based ratios
        ratios = {class_id: 1 / acc if acc > 0 else float('inf') for class_id, acc in avg_class_accuracies.items()}
        return ratios

    @staticmethod
    def compute_sample_allocation(ratios, desired_dataset_size):
        # Normalize ratios to sum to 1
        total_ratio = sum(ratios.values())
        normalized_ratios = {class_id: ratio / total_ratio for class_id, ratio in ratios.items()}

        # Allocate samples based on normalized ratios
        samples_per_class = {class_id: int(round(normalized_ratio * desired_dataset_size))
                             for class_id, normalized_ratio in normalized_ratios.items()}

        # Adjust to ensure the total matches desired_dataset_size (due to rounding errors)
        total_allocated = sum(samples_per_class.values())
        if total_allocated != desired_dataset_size:
            difference = desired_dataset_size - total_allocated
            sorted_classes = sorted(samples_per_class.keys(), key=lambda cid: -ratios[cid])
            for class_id in sorted_classes:
                samples_per_class[class_id] += 1 if difference > 0 else -1
                difference += -1 if difference > 0 else 1
                if difference == 0:
                    break

        return samples_per_class

    @staticmethod
    def print_ratios_and_allocation(ratios, samples_per_class):
        print("\nHardness-Based Ratios and Sample Allocation:")
        for class_id, ratio in ratios.items():
            print(f"  Class {class_id}: Ratio = {ratio:.2f}, Samples = {samples_per_class[class_id]}")

    def main(self):
        self.load_results()
        avg_dataset_accuracy, avg_class_accuracies, num_models = self.compute_average_accuracies()
        self.print_results(avg_dataset_accuracy, avg_class_accuracies, num_models)

        # Compute hardness-based ratios
        ratios = self.compute_hardness_based_ratios(avg_class_accuracies)

        # Compute sample allocation
        samples_per_class = self.compute_sample_allocation(ratios, self.desired_dataset_size)

        # Print results
        self.print_ratios_and_allocation(ratios, samples_per_class)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and compute average accuracies from saved results.")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="Name of the dataset (e.g., CIFAR10 or CIFAR100)")
    parser.add_argument('--desired_dataset_size', type=int, required=True,
                        help="Desired size of the dataset after rebalancing")
    args = parser.parse_args()

    experiment = Experiment3(args.dataset_name, args.desired_dataset_size)
    experiment.main()
