import pickle
import os
import argparse
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import get_config
from data_pruning import DataResampling  # Assuming DataResampling is in a separate file


class Experiment3:
    def __init__(self, dataset_name, desired_dataset_size):
        self.dataset_name = dataset_name
        self.desired_dataset_size = desired_dataset_size
        self.results_file = os.path.join('Results', dataset_name, 'el2n_scores.pkl')
        self.config = get_config(dataset_name)
        self.num_classes = self.config['num_classes']

        # Reproducibility settings
        self.seed = 42
        self.set_reproducibility()

    def set_reproducibility(self):
        """
        Ensure reproducibility by setting seeds and configuring PyTorch settings.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_dataset(self):
        """
        Load the dataset based on dataset_name.
        """
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(self.config['mean'], self.config['std']),
        ])

        if self.dataset_name == 'CIFAR10':
            return datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        elif self.dataset_name == 'CIFAR100':
            return datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        elif self.dataset_name == 'SVHN':
            return datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")

    def load_results(self):
        """
        Load the computed accuracies from results_file.
        """
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        with open(self.results_file, 'rb') as file:
            return pickle.load(file)

    def compute_hardness_based_ratios(self, class_accuracies):
        """
        Compute hardness-based ratios based on class-level accuracies.
        """
        class_accumulator = {i: 0 for i in range(self.num_classes)}
        class_counts = {i: 0 for i in range(self.num_classes)}

        for model_class_acc in class_accuracies:
            for class_id, acc in model_class_acc.items():
                class_accumulator[class_id] += acc
                class_counts[class_id] += 1

        avg_class_accuracies = {class_id: class_accumulator[class_id] / class_counts[class_id]
                                for class_id in class_accumulator}

        # Compute ratios as 1 / accuracy
        return {class_id: 1 / acc if acc > 0 else float('inf') for class_id, acc in avg_class_accuracies.items()}

    def compute_sample_allocation(self, ratios):
        """
        Compute the number of samples required for each class to match the desired_dataset_size.
        """
        total_ratio = sum(ratios.values())
        normalized_ratios = {class_id: ratio / total_ratio for class_id, ratio in ratios.items()}

        # Allocate samples based on normalized ratios
        samples_per_class = {class_id: int(round(normalized_ratio * self.desired_dataset_size))
                             for class_id, normalized_ratio in normalized_ratios.items()}

        # Adjust to ensure total matches desired_dataset_size
        total_allocated = sum(samples_per_class.values())
        if total_allocated != self.desired_dataset_size:
            difference = self.desired_dataset_size - total_allocated
            sorted_classes = sorted(samples_per_class.keys(), key=lambda cid: -ratios[cid])
            for class_id in sorted_classes:
                samples_per_class[class_id] += 1 if difference > 0 else -1
                difference += -1 if difference > 0 else 1
                if difference == 0:
                    break

        return samples_per_class

    def resample_dataset(self, dataset, samples_per_class):
        """
        Use DataResampling to modify the dataset to match the samples_per_class.
        """
        resampler = DataResampling(dataset, self.num_classes)
        return resampler.resample_data(samples_per_class)

    def get_dataloader(self, dataset):
        """
        Create a DataLoader with deterministic worker initialization.
        """
        def worker_init_fn(worker_id):
            np.random.seed(self.seed + worker_id)
            random.seed(self.seed + worker_id)

        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True,
                          num_workers=2, worker_init_fn=worker_init_fn)

    def main(self):
        # Load dataset and results
        dataset = self.load_dataset()
        _, _, _, dataset_accuracies, class_accuracies = self.load_results()

        # Compute hardness-based ratios and sample allocation
        ratios = self.compute_hardness_based_ratios(class_accuracies)
        samples_per_class = self.compute_sample_allocation(ratios)

        # Perform resampling
        resampled_dataset = self.resample_dataset(dataset, samples_per_class)

        # Get DataLoader for resampled dataset
        resampled_loader = self.get_dataloader(resampled_dataset)

        # Print final sample allocation
        print("Final Samples Per Class After Resampling:")
        for class_id, count in samples_per_class.items():
            print(f"  Class {class_id}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment3 with Data Resampling.")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="Name of the dataset (e.g., CIFAR10, CIFAR100, SVHN)")
    parser.add_argument('--desired_dataset_size', type=int, required=True,
                        help="Desired size of the dataset after resampling")
    args = parser.parse_args()

    experiment = Experiment3(args.dataset_name, args.desired_dataset_size)
    experiment.main()
