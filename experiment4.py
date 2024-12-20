import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from neural_networks import ResNet18LowRes
from utils import get_config

# Fetch CIFAR-10 configuration
config = get_config('CIFAR10')

# Constants from config
NUM_CLASSES = 2  # Binary classification (classes 2 -> 0, 4 -> 1)
BATCH_SIZE = config['batch_size']
LR = config['lr']
MOMENTUM = config['momentum']
WEIGHT_DECAY = config['weight_decay']
NUM_EPOCHS = 200  # As specified
NUM_MODELS = 20  # Number of networks in the ensemble
CLASS_FILTER = [2, 4]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations using the configuration
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(config['mean'], config['std'])
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


# Filter and relabel datasets
def filter_and_relabel(dataset, class_filter):
    indices = [i for i, label in enumerate(dataset.targets) if label in class_filter]
    relabeled_targets = [0 if dataset.targets[i] == class_filter[0] else 1 for i in indices]

    # Create a new dataset with filtered images and relabeled targets
    dataset.data = dataset.data[indices]
    dataset.targets = relabeled_targets

    return dataset


train_dataset_filtered = filter_and_relabel(train_dataset, CLASS_FILTER)
test_dataset_filtered = filter_and_relabel(test_dataset, CLASS_FILTER)

# Create dataloaders
train_loader = DataLoader(train_dataset_filtered, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset_filtered, batch_size=BATCH_SIZE, shuffle=False)


# Function to compute accuracy per class
def compute_accuracy_per_class(model, test_loader, num_classes, device):
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for label, prediction in zip(labels, predicted):
                class_total[label] += 1
                if label == prediction:
                    class_correct[label] += 1
                total_correct += (prediction == label).item()
                total_samples += 1

    class_accuracies = [
        100 * (class_correct[i] / class_total[i]) if class_total[i] > 0 else 0
        for i in range(num_classes)
    ]
    overall_accuracy = 100 * total_correct / total_samples
    return overall_accuracy, class_accuracies


# Function to train a single model
def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Compute accuracy after each epoch
        overall_accuracy, class_accuracies = compute_accuracy_per_class(model, test_loader, NUM_CLASSES, DEVICE)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        for class_idx, class_acc in enumerate(class_accuracies):
            print(f"Class {class_idx} Accuracy: {class_acc:.2f}%")


# Training an ensemble of 20 models
ensemble = []
for model_idx in range(NUM_MODELS):
    print(f"Training model {model_idx + 1}/{NUM_MODELS}")
    model = ResNet18LowRes(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, test_loader, optimizer, criterion, NUM_EPOCHS, DEVICE)
    # Save the trained model
    ensemble.append(model)
    torch.save(model.state_dict(), f"{config['save_dir']}model_{model_idx + 1}.pth")

print("Ensemble training completed.")
