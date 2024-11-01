import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from neural_networks import ResNet18LowRes
from utils import get_config

class HardnessCalculator:
    def __init__(self, dataset_name):
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.dataset_name = dataset_name
        config = get_config(dataset_name)
        self.BATCH_SIZE = config['batch_size']
        self.SAVE_EPOCH = config['save_epoch']
        self.MODEL_DIR = config['save_dir']
        self.NUM_CLASSES = config['num_classes']

        self.training_loader, _, self.training_set_size = self.load_dataset(self.dataset_name)

    def load_dataset(self, dataset_name):
        if dataset_name == 'CIFAR10':
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                        transform=train_transform)
        elif dataset_name == 'CIFAR100':
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            training_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                         transform=train_transform)
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        training_loader = DataLoader(training_set, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=2)
        return training_loader, None, len(training_set)

    def create_model(self):
        model = ResNet18LowRes(num_classes=self.NUM_CLASSES)
        return model

    def compute_el2n(self, model, dataloader):
        model.eval()
        el2n_scores = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                softmax_outputs = F.softmax(outputs, dim=1)
                one_hot_labels = F.one_hot(labels, num_classes=self.NUM_CLASSES).float()
                l2_errors = torch.norm(softmax_outputs - one_hot_labels, dim=1)
                el2n_scores.extend(l2_errors.cpu().numpy())
        return el2n_scores

    def load_model_and_compute_el2n(self, model_id):
        model = self.create_model().cuda()
        model_path = os.path.join(self.MODEL_DIR, 'none', self.dataset_name,
                                  f'model_{model_id}_epoch_{self.SAVE_EPOCH}.pth')

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            el2n_scores = self.compute_el2n(model, self.training_loader)
            return el2n_scores
        else:
            print(f'Model {model_id} not found at epoch {self.SAVE_EPOCH}.')
            return None

    def collect_el2n_scores(self):
        all_el2n_scores = [[] for _ in range(self.training_set_size)]
        for model_id in range(10):
            el2n_scores = self.load_model_and_compute_el2n(model_id)
            if el2n_scores:
                for i in range(self.training_set_size):
                    all_el2n_scores[i].append(el2n_scores[i])
        return all_el2n_scores

    def group_scores_by_class(self, el2n_scores):
        class_el2n_scores = {i: [] for i in range(self.NUM_CLASSES)}  # Dictionary to store scores by class
        labels = []  # Store corresponding labels

        # Since we are not shuffling the data loader, we can directly match scores with their labels
        for i, (_, label) in enumerate(self.training_loader.dataset):
            class_el2n_scores[label].append(el2n_scores[i])
            labels.append(label)  # Collect the labels

        return class_el2n_scores, labels

    @staticmethod
    def compute_class_statistics(class_el2n_scores):
        class_stats = {}

        for class_id, scores in class_el2n_scores.items():
            scores_array = np.array(scores)  # Convert to numpy array for statistical analysis

            # Compute mean, std, and quartiles
            means = np.mean(scores_array, axis=1)
            q1 = np.percentile(means, 25)
            q3 = np.percentile(means, 75)
            min_val = np.min(means)
            max_val = np.max(means)

            class_stats[class_id] = {
                "q1": q1,
                "q3": q3,
                "min": min_val,
                "max": max_val
            }

        return class_stats

    def plot_class_level_candlestick(self, class_stats):
        # Prepare the saving directory and file name
        save_dir = os.path.join('Figures/', self.dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f'hardness_distribution.pdf')

        # Prepare the data for plotting
        class_ids = list(class_stats.keys())
        q1_values = [class_stats[class_id]["q1"] for class_id in class_ids]
        q3_values = [class_stats[class_id]["q3"] for class_id in class_ids]
        min_values = [class_stats[class_id]["min"] for class_id in class_ids]
        max_values = [class_stats[class_id]["max"] for class_id in class_ids]

        # Create the candlestick chart
        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(self.NUM_CLASSES):
            # Draw the candlestick (real body: Q1 to Q3, shadow: min to max)
            ax.plot([i, i], [min_values[i], max_values[i]], color='black')  # Shadow
            ax.plot([i, i], [q1_values[i], q3_values[i]], color='blue', lw=6)  # Real body

        ax.set_xticks(range(self.NUM_CLASSES))
        ax.set_xticklabels([f'Class {i}' for i in range(self.NUM_CLASSES)])
        ax.set_xlabel("Classes")
        ax.set_ylabel("EL2N Score (L2 Norm)")
        ax.set_title("Class-Level EL2N Scores Candlestick Plot")
        plt.savefig(file_name)

    def save_el2n_scores(self, el2n_scores):
        np.save(f'{self.dataset_name}_el2n_scores.npy', el2n_scores)

    def run(self):
        all_el2n_scores = self.collect_el2n_scores()
        class_el2n_scores, labels = self.group_scores_by_class(all_el2n_scores)
        self.save_el2n_scores((all_el2n_scores, class_el2n_scores, labels))
        print("Hardness scores computed and saved. Now producing to produce visualization Figures.")
        class_stats = self.compute_class_statistics(class_el2n_scores)
        self.plot_class_level_candlestick(class_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Hardness of Dataset Samples')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10', help='Specify the dataset name (default: CIFAR10)')
    args = parser.parse_args()

    calculator = HardnessCalculator(args.dataset_name)
    calculator.run()
