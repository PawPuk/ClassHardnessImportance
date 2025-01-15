import argparse
import os
import pickle

import torch
import torch.nn.functional as F

from neural_networks import ResNet18LowRes
from utils import get_config, load_dataset

# Measure the overlap between the pruned subsets as a function of hardness estimator and threshold (percentage of data
# pruned). Present results in a 2D heatmap with y being the threshold and x being the ensemble size (overlap between
# the subset pruned by ensemble with x_i models and x_{i+1} models). This should give 3 heatmaps.

def create_model():
    model = ResNet18LowRes(num_classes=NUM_CLASSES)
    return model


def compute_el2n(model, dataloader):
    model.eval()
    el2n_scores = []

    # Accuracy is another way to estimate class-level hardness, so we also compute it
    class_correct = {i: 0 for i in range(NUM_CLASSES)}
    class_total = {i: 0 for i in range(NUM_CLASSES)}

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            # This computes the EL2N scores
            softmax_outputs = F.softmax(outputs, dim=1)
            one_hot_labels = F.one_hot(labels, num_classes=NUM_CLASSES).float()
            l2_errors = torch.norm(softmax_outputs - one_hot_labels, dim=1)
            el2n_scores.extend(l2_errors.cpu().numpy())

            # This computes the class-level accuracies
            _, predicted = torch.max(outputs, 1)
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    class_accuracies = {k: class_correct[k] / class_total[k] if class_total[k] > 0 else 0 for k in class_correct}

    return el2n_scores, class_accuracies


def collect_el2n_scores(loader, n):
    all_el2n_scores, model_class_accuracies = [[] for _ in range(n)], []
    for model_id in range(NUM_MODELS):
        model = create_model().cuda()
        # This is the directory in which we store the pretrained models.
        model_path = os.path.join(MODEL_DIR, 'none', f"{args.remove_noise}{args.dataset_name}",
                                  f'model_{model_id}_epoch_{SAVE_EPOCH}.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            el2n_scores, class_accuracies = compute_el2n(model, loader)
        else:
            # This code can only be run if models were pretrained. If no pretrained models are found, throw error.
            raise Exception(f'Model {model_id} not found at epoch {SAVE_EPOCH}.')
        for i in range(n):
            all_el2n_scores[i].append(el2n_scores[i])
        model_class_accuracies.append(class_accuracies)
    return all_el2n_scores, model_class_accuracies


def save_el2n_scores(el2n_scores):
    with open(os.path.join(RESULTS_SAVE_DIR, 'el2n_scores.pkl'), 'wb') as file:
        pickle.dump(el2n_scores, file)


def load_results(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def main():
    training_loader, training_set_size, _, _ = load_dataset(args.dataset_name, args.remove_noise, SEED)
    training_all_el2n_scores, training_class_accuracies = collect_el2n_scores(training_loader, training_set_size)
    save_el2n_scores((training_all_el2n_scores, training_class_accuracies))

    aum_path = os.path.join(HARDNESS_SAVE_DIR, 'AUM.pkl')
    forgetting_path = os.path.join(HARDNESS_SAVE_DIR, 'Forgetting.pkl')
    aum_scores = load_results(aum_path)
    forgetting_scores = load_results(forgetting_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Hardness of Dataset Samples')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='Specify the dataset name (default: CIFAR10)')
    parser.add_argument('--remove_noise', action='store_true', help='Raise this flag to remove noise from the data.')
    args = parser.parse_args()

    NUM_CLASSES = get_config(args.dataset_name)['num_classes']
    NUM_MODELS = get_config(args.dataset_name)['num_models']
    MODEL_DIR = get_config(args.dataset_name)['save_dir']
    SAVE_EPOCH = get_config(args.dataset_name)['save_epoch']
    RESULTS_SAVE_DIR = os.path.join('Results/', f"{args.remove_noise}{args.dataset_name}")
    SEED = 42
    HARDNESS_SAVE_DIR = f"Results/{args.remove_noise}{args.dataset_name}/"

    main()