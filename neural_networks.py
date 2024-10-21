import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18LowRes(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18LowRes, self).__init__()

        # Load the standard ResNet18 model from torchvision, but don't use its first conv layer
        resnet = models.resnet18(num_classes=num_classes)

        # Replace the first two layers (7x7 Conv and MaxPool) with a 3x3 Conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Use the rest of the resnet layers as is
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Adjust the final fully connected layer for the number of classes (e.g., CIFAR-10 or CIFAR-100)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Modified first convolution layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Continue with the rest of ResNet18
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # Flatten the tensor for the fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x