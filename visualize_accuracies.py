import torch
from torchvision.models import resnet18
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt

# Define constants
BATCH_SIZE = 128
NUM_CLASSES = 10
MODEL_DIR = './Exp1Models/'  # Directory where the trained models are saved
NUM_MODELS = 10  # Number of models in the ensemble

# Transformations for the test set (no augmentations, just normalization)
test_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 Dataset (Test set)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# Function to load a saved model
def load_model(model_id):
    model = resnet18(num_classes=NUM_CLASSES).cuda()
    model_path = os.path.join(MODEL_DIR, f'model_{model_id}_epoch_200.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Function to compute overall and class-level accuracies
def compute_accuracies(model, dataloader):
    class_correct = np.zeros(NUM_CLASSES)
    class_total = np.zeros(NUM_CLASSES)
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Calculate per-class correct predictions
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    overall_accuracy = total_correct / total_samples  # Overall dataset-level accuracy
    class_accuracies = class_correct / class_total  # Class-level accuracies

    return overall_accuracy, class_accuracies


# Function to plot class-level accuracies using candlesticks and dataset-level accuracy with fill_between
def plot_accuracies(overall_accuracies, class_accuracies_list):
    classes = [f'Class {i}' for i in range(NUM_CLASSES)]
    x = np.arange(NUM_CLASSES)

    # Compute mean and std for class-level accuracies
    mean_class_accuracies = np.mean(class_accuracies_list, axis=0)
    std_class_accuracies = np.std(class_accuracies_list, axis=0)
    min_class_accuracies = np.min(class_accuracies_list, axis=0)
    max_class_accuracies = np.max(class_accuracies_list, axis=0)

    # Compute mean and std for overall accuracy
    mean_overall_accuracy = np.mean(overall_accuracies)
    std_overall_accuracy = np.std(overall_accuracies)

    plt.figure(figsize=(10, 6))

    # Plot class-level accuracies as candlesticks
    plt.errorbar(x, mean_class_accuracies, yerr=[mean_class_accuracies - min_class_accuracies, max_class_accuracies - mean_class_accuracies],
                 fmt='o', color='blue', ecolor='black', elinewidth=2, capsize=5, label='Class-level Accuracies')

    # Plot overall dataset-level accuracy with fill_between for mean ± std
    plt.fill_between(
        [-0.5, NUM_CLASSES - 0.5],  # X range
        mean_overall_accuracy - std_overall_accuracy,  # Lower bound (mean - std)
        mean_overall_accuracy + std_overall_accuracy,  # Upper bound (mean + std)
        color='red', alpha=0.3, label=f'Dataset-level Accuracy (Mean ± Std)'
    )
    plt.axhline(y=mean_overall_accuracy, color='red', linestyle='--', label='Dataset-level Mean Accuracy')

    plt.xticks(x, classes)
    plt.ylabel('Accuracy')
    plt.title('Class-level Accuracies with Candlesticks and Dataset-level Accuracy with Fill')
    plt.legend()
    plt.savefig('Accuracies.pdf')
    plt.show()


# Main script
if __name__ == '__main__':
    overall_accuracies = []
    class_accuracies_list = []

    # Loop through all saved models in the ensemble
    for model_id in range(NUM_MODELS):
        print(f"Evaluating Model {model_id}...")
        model = load_model(model_id)

        # Compute the accuracies for this model
        overall_accuracy, class_accuracies = compute_accuracies(model, testloader)

        # Store the accuracies
        overall_accuracies.append(overall_accuracy)
        class_accuracies_list.append(class_accuracies)

    # Convert lists to numpy arrays for easier handling
    overall_accuracies = np.array(overall_accuracies)
    class_accuracies_list = np.array(class_accuracies_list)

    # Plot the accuracies
    plot_accuracies(overall_accuracies, class_accuracies_list)
