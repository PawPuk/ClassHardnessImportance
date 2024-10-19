import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

import utils as u

# Training settings
def train_network(local_rank):
    dist.init_process_group(backend="nccl", init_method="env://", world_size=u.GPUS_PER_NODE, rank=local_rank)

    # Load the CIFAR-100 dataset
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    trainset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    valset = CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Batch size per GPU
    batch_size = u.CONFIG['batch_size'] // u.GPUS_PER_NODE

    # Dataloaders for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=u.GPUS_PER_NODE,
                                                                    rank=local_rank)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Define model, optimizer, criterion, scheduler
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_0", pretrained=False).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss().to(local_rank)

    # Scale the learning rate based on the batch size and world size
    base_lr = u.CONFIG['lr']
    basic_bs = u.CONFIG['basic_bs']
    lr_scaled = base_lr * (u.CONFIG['batch_size'] * u.GPUS_PER_NODE / basic_bs)

    optimizer = optim.SGD(model.parameters(), lr=lr_scaled, momentum=u.CONFIG['momentum'],
                          dampening=u.CONFIG['dampening'], weight_decay=u.CONFIG['weight_decay'],
                          nesterov=u.CONFIG['nesterov'])

    scheduler = CosineAnnealingLR(optimizer, T_max=u.CONFIG['max_epochs'], eta_min=u.CONFIG['eta_min'])
    cudnn.benchmark = True

    # Training loop
    for epoch in tqdm(range(u.CONFIG['max_epochs'])):
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
            if i % u.CONFIG['log_interval'] == 0 and local_rank == 0:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / (i + 1):.4f}")

        scheduler.step()

        # Validation step (optional)
        if epoch % u.CONFIG['eval_interval'] == 0:
            validate(model, val_loader, criterion, local_rank)

        # Save model parameters every 2 epochs, moved to CPU for later use on non-GPU machines
        if (epoch + 1) % 2 == 0 and local_rank == 0:  # Save only on rank 0
            checkpoint_path = f"{u.MODELS_DIR}model_epoch_{epoch + 1}.pth"
            model_cpu_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(model_cpu_state_dict, checkpoint_path)
            print(f"Model parameters saved at {checkpoint_path}")

    print('Training complete')


def validate(model, val_loader, criterion, local_rank):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
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
    # Spawn processes for distributed training
    torch.multiprocessing.spawn(train_network, nprocs=u.GPUS_PER_NODE)


