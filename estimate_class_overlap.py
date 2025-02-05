from multiprocessing import Pool, cpu_count, Manager, Lock
import os
from tqdm import tqdm

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torchvision
import torchvision.transforms as transforms
import itertools

# 1. Load CIFAR-100 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# 2. Function to extract data for specific classes
def get_data_for_classes(class1, class2, dataset):
    """Extracts data for two given classes, assigning labels (0,1)."""
    X, y = [], []
    for data, label in dataset:
        if label in [class1, class2]:
            X.append(data.view(-1).numpy())  # Flatten image
            y.append(0 if label == class1 else 1)  # Binary labels for OvO classification

    X = np.array(X)
    y = np.array(y)

    # Normalize features (important for SVM)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# 3. Load or Initialize Accuracy Dictionary
accuracy_file = 'ovo_accuracies.npy'
if os.path.exists(accuracy_file):
    print("Accuracies file found. Loading existing progress...")
    loaded_accuracies = np.load(accuracy_file, allow_pickle=True).item()
else:
    print("No accuracy file found. Starting fresh training...")
    loaded_accuracies = {i: [] for i in range(100)}

# 4. Identify completed class pairs
completed_pairs = set()
for class_idx in loaded_accuracies:
    for class_pair in loaded_accuracies[class_idx]:
        completed_pairs.add((class_idx, class_pair[0]))

# 5. Define Training Function (Runs in Parallel)
def train_ovo_lsvc(class_pair, shared_accuracies, lock):
    """Trains an LSVC model for a given class pair and returns the result."""
    class1, class2 = class_pair

    if (class1, class2) in completed_pairs or (class2, class1) in completed_pairs:
        return None

    print(f"Training LSVC for class pair ({class1}, {class2})...")

    X_train, y_train = get_data_for_classes(class1, class2, trainset)
    X_test, y_test = get_data_for_classes(class1, class2, testset)

    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    with lock:
        shared_accuracies[class1].append((class2, test_accuracy))
        shared_accuracies[class2].append((class1, test_accuracy))

        # Save after every model is trained
        np.save(accuracy_file, dict(shared_accuracies))

    return class1, class2, test_accuracy

# 6. Set Up Shared Memory & Parallel Training
manager = Manager()
shared_accuracies = manager.dict(loaded_accuracies)  # Shared dictionary between processes
lock = manager.Lock()  # Lock to prevent simultaneous file writes

all_class_pairs = list(itertools.combinations(range(100), 2))
remaining_pairs = [pair for pair in all_class_pairs
                   if pair not in completed_pairs and (pair[1], pair[0]) not in completed_pairs]

print(f"Starting parallel training with {cpu_count()} CPU cores...")
def worker_train_ovo_lsvc(pair):
    """ Wrapper function to call train_ovo_lsvc with shared variables. """
    return train_ovo_lsvc(pair, shared_accuracies, lock)

# Use this instead of the lambda function:
with Pool(processes=cpu_count()) as pool:
    results = list(tqdm(pool.imap(worker_train_ovo_lsvc, remaining_pairs), total=len(remaining_pairs)))


# 7. Final Save
np.save(accuracy_file, dict(shared_accuracies))
print(f"Final results saved to {accuracy_file}")
