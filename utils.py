import os

import torch


CONFIG = {
    'batch_size': 256,  # Global batch size across all GPUs
    'basic_bs': 256,  # Base batch size used to calculate scaled LR
    'lr': 0.1,  # Base learning rate (before scaling)
    'momentum': 0.9,  # Momentum for SGD
    'dampening': 0,  # Dampening for momentum
    'weight_decay': 5e-4,  # Weight decay (L2 regularization)
    'nesterov': True,  # Use Nesterov momentum
    'max_epochs': 200,  # Number of training epochs
    'eta_min': 0,  # Minimum learning rate for CosineAnnealingLR
    'log_interval': 100,  # How often to log training progress
    'eval_interval': 10  # How often to evaluate on validation set
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPUS_PER_NODE = torch.cuda.device_count()
cpus_per_node = os.cpu_count()
if DEVICE == 'cpu' or GPUS_PER_NODE == 0:
    NUM_WORKERS = cpus_per_node
else:
    NUM_WORKERS = cpus_per_node // GPUS_PER_NODE

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

MODELS_DIR = 'Models/'
CORRELATION_FIGURES_DIR = 'CorrelationFigures/'
DISTRIBUTION_FIGURES_DIR = 'DistributionFigures/'

for directory in [CORRELATION_FIGURES_DIR, DISTRIBUTION_FIGURES_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)
