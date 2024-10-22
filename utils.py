dataset_configs = {
    'CIFAR10': {
        'num_epochs': 200,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_decay_milestones': [60, 120, 160],
        'save_epoch': 20,
        'save_dir': './Models/',
        'timings_dir': './Timings/',
        'num_models': 10,
        'num_classes': 10
    },
    'CIFAR100': {
        'num_epochs': 200,
        'lr': 0.05,
        'momentum': 0.85,
        'weight_decay': 0.0001,
        'lr_decay_milestones': [50, 150, 225],
        'save_epoch': 25,
        'save_dir': './Models/',
        'timings_dir': './Timings/',
        'num_models': 8,
        'num_classes': 100
    }
}


def get_config(dataset_name):
    """
    Fetch the appropriate configuration based on the dataset name.
    """
    if dataset_name in dataset_configs:
        return dataset_configs[dataset_name]
    else:
        raise ValueError(f"Configuration for dataset {dataset_name} not found!")
