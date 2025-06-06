import os

ROOT = '.'

dataset_configs = {
    'CIFAR10': {
        'batch_size': 128,
        'num_epochs': 200,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_decay_milestones': [60, 120, 160],
        'save_epoch': 20,
        'save_dir': os.path.join(ROOT, 'Models/'),
        'timings_dir': os.path.join(ROOT, 'Timings/'),
        'num_models': 20,
        'num_classes': 10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
        'num_training_samples': [5000 for _ in range(10)],
        'num_test_samples': [1000 for _ in range(10)],
        'safe_pruning_ratios': [48.8, 74.22, 29.08, 10.64, 45.8, 31.44, 54.92, 58.36, 67.28, 63.66],
        'robust_ensemble_size': 8,  # This was set manually after the robustness analysis
        'class_names': ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    },
    'CIFAR100': {
        'batch_size': 128,
        'num_epochs': 200,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_decay_milestones': [60, 120, 160],
        'save_epoch': 20,
        'save_dir': os.path.join(ROOT, 'Models/'),
        'timings_dir': os.path.join(ROOT, 'Timings/'),
        'num_models': 20,
        'num_classes': 100,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
        'num_training_samples': [500 for _ in range(100)],
        'num_test_samples': [100 for _ in range(100)],
        'robust_ensemble_size': 8,  # This was set manually after the robustness analysis
        'class_names': [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
            'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
            'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
            'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
            'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    }
}


def get_config(dataset_name):
    """
    Fetch the appropriate configuration based on the dataset name.
    """
    if dataset_name in dataset_configs:
        config = dataset_configs[dataset_name]
        config['probe_base_seed'] = 42
        config['probe_seed_step'] = 42
        return config
    else:
        raise ValueError(f"Configuration for dataset {dataset_name} not found!")
