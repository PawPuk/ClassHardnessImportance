import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# 1. Load CIFAR-100 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Common normalization for CIFAR-100
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)


# 2. Prepare the data for SVM
def get_data_for_class(class_idx, dataset):
    """
    Prepare binary labels where class_idx is positive (1) and all others are negative (0).
    """
    X = []
    y = []
    for data, label in dataset:
        # Flatten the image tensor to 1D
        X.append(data.view(-1).numpy())
        y.append(1 if label == class_idx else 0)
    X = np.array(X)
    y = np.array(y)

    # Normalize features (important for SVM)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


# 3. Check if accuracies are already computed
accuracy_file = 'accuracies.npy'

if os.path.exists(accuracy_file):
    print("Accuracies found. Loading from file...")
    # Load the saved accuracies
    accuracies = np.load(accuracy_file, allow_pickle=True)
    train_accuracies = accuracies.item().get('train_accuracies')
    test_accuracies = accuracies.item().get('test_accuracies')
else:
    print("Accuracies not found. Training LSVCs...")
    train_accuracies = []
    test_accuracies = []

    # Train an LSVC for each class and track accuracy
    for class_idx in range(100):  # There are 100 classes in CIFAR-100
        print(f"Training for class {class_idx}...")

        # Prepare training data for the current class
        X_train, y_train = get_data_for_class(class_idx, trainset)
        X_test, y_test = get_data_for_class(class_idx, testset)

        # Train a Linear SVM
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)

        # Test the model on training set
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Test the model on test set
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Append the accuracies
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f"Training accuracy for class {class_idx}: {train_accuracy:.4f}")
        print(f"Test accuracy for class {class_idx}: {test_accuracy:.4f}")

    # Save the accuracies to a file
    accuracies = {'train_accuracies': train_accuracies, 'test_accuracies': test_accuracies}
    np.save(accuracy_file, accuracies)
    print(f"Accuracies saved to {accuracy_file}")

# 4. Plot the accuracies
plt.figure(figsize=(10, 5))

# Plot the training accuracy
plt.plot(range(100), train_accuracies, label='Training Accuracy', color='blue', linestyle='-', marker='o')

# Plot the test accuracy
plt.plot(range(100), test_accuracies, label='Test Accuracy', color='red', linestyle='--', marker='x')

# Add labels and title
plt.xlabel('Class Index')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy for Each Class (CIFAR-100)')
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()
