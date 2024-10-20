import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os

# Define constants
BATCH_SIZE = 128
NUM_EPOCHS = 200
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
SAVE_EPOCH = 20  # Save state after 20 epochs
LR_DECAY_MILESTONES = [60, 120, 160]  # Learning rate schedule

# Transformations for CIFAR-10 dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# Define the ResNet-18 model
def create_model():
    model = resnet18(num_classes=10)
    return model


# Function to evaluate accuracy and loss on the test set
def evaluate_model(model, dataloader, criterion):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for testing
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)  # Multiply by batch size
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / total
    return avg_loss, accuracy


# Training function
def train_model(model, optimizer, criterion, trainloader, testloader, scheduler, num_epochs, save_dir, model_id):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)  # Multiply by batch size
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Compute average loss and accuracy for training set
        avg_train_loss = running_loss / total_train
        train_accuracy = 100 * correct_train / total_train

        # Evaluate the model on the test set
        avg_test_loss, test_accuracy = evaluate_model(model, testloader, criterion)

        print(f'Model {model_id}, Epoch [{epoch + 1}/{num_epochs}] '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

        # Save model at epoch 20 and final epoch
        if epoch + 1 == SAVE_EPOCH or epoch + 1 == num_epochs:
            save_path = os.path.join(save_dir, f'model_{model_id}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'Model {model_id} saved at epoch {epoch + 1}')

        # Step the scheduler at the end of the epoch
        scheduler.step()


# Main script
if __name__ == '__main__':
    # Ensure directory for saving models exists
    save_dir = './Exp1Models/'
    os.makedirs(save_dir, exist_ok=True)

    # Loop over 10 independent initializations
    for model_id in range(10):
        # Initialize the model, loss function, optimizer, and learning rate scheduler
        model = create_model().cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_DECAY_MILESTONES, gamma=0.2)

        # Train the model and save its state
        train_model(model, optimizer, criterion, trainloader, testloader, scheduler, NUM_EPOCHS, save_dir, model_id)
