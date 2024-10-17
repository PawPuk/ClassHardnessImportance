import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import os
from tqdm import tqdm


# Configuration settings
config = {
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


# Training settings
def train_network(local_rank, ngpus_per_node, config):
    dist.init_process_group(backend="nccl", init_method="env://", world_size=ngpus_per_node, rank=local_rank)

    # Load the CIFAR-100 dataset
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    trainset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    valset = CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Batch size per GPU
    batch_size = config['batch_size'] // ngpus_per_node

    # Dataloaders for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=ngpus_per_node,
                                                                    rank=local_rank)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Define model, optimizer, criterion, scheduler
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_0", pretrained=False).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss().to(local_rank)

    # Scale the learning rate based on the batch size and world size
    base_lr = config['lr']
    basic_bs = config['basic_bs']
    lr_scaled = base_lr * (config['batch_size'] * ngpus_per_node / basic_bs)

    optimizer = optim.SGD(model.parameters(), lr=lr_scaled, momentum=config['momentum'],
                          dampening=config['dampening'], weight_decay=config['weight_decay'],
                          nesterov=config['nesterov'])

    scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epochs'], eta_min=config['eta_min'])

    cudnn.benchmark = True

    # Training loop
    for epoch in tqdm(range(config['max_epochs'])):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(local_rank), labels.to(local_rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % config['log_interval'] == 0 and local_rank == 0:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / (i + 1):.4f}")

        scheduler.step()

        # Validation step (optional)
        if epoch % config['eval_interval'] == 0:
            validate(model, val_loader, criterion, local_rank)

    print('Training complete')


def validate(model, val_loader, criterion, local_rank):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(local_rank), labels.to(local_rank)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    ngpus_per_node = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Spawn processes for distributed training
    torch.multiprocessing.spawn(train_network, nprocs=ngpus_per_node, args=(ngpus_per_node, config))

