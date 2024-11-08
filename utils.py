dataset_configs = {
    'CIFAR10': {
        'batch_size': 128,
        'num_epochs': 200,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_decay_milestones': [60, 120, 160],
        'save_epoch': 20,
        'save_dir': './Models/',
        'timings_dir': './Timings/',
        'num_models': 10,
        'num_classes': 10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
        'num_training_samples': 50000
    },
    'CIFAR100': {
        'batch_size': 128,
        'num_epochs': 200,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_decay_milestones': [60, 120, 160],
        'save_epoch': 20,
        'save_dir': './Models/',
        'timings_dir': './Timings/',
        'num_models': 10,
        'num_classes': 100,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
        'num_training_samples': 50000
    }
}

"""Other setting for CIFAR100 proposed by ChatGPT that appears to give better results
'CIFAR100': {
        'batch_size': 128,
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
"""


def get_config(dataset_name):
    """
    Fetch the appropriate configuration based on the dataset name.
    """
    if dataset_name in dataset_configs:
        config = dataset_configs[dataset_name]
        config['probe_base_seed'] = 42
        config['probe_seed_step'] = 42
        config['new_base_seed'] = 4242
        config['new_seed_step'] = 42
        return config
    else:
        raise ValueError(f"Configuration for dataset {dataset_name} not found!")
