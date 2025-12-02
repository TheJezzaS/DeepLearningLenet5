import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))# Normalize to range [-1, 1]
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

class LeNet5(nn.Module):
    def __init__(self, use_dropout=False, use_bn=False):
        super(LeNet5, self).__init__()
        self.use_dropout = use_dropout
        self.use_bn = use_bn

        # Convolution 1: Input 1x32x32 -> Output 6x28x28
        self.conv1 = nn.Conv2d(1, 6, 5)
        if use_bn: self.bn1 = nn.BatchNorm2d(6)

        # Pooling 1: Input 6x28x28 -> Output 6x14x14
        self.pool = nn.MaxPool2d(2, 2)

        # Convolution 2: Input 6x14x14 -> Output 16x10x10
        self.conv2 = nn.Conv2d(6, 16, 5)
        if use_bn: self.bn2 = nn.BatchNorm2d(16)

        # Pooling 2: Input 16x10x10 -> Output 16x5x5

        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        if use_bn: self.bn3 = nn.BatchNorm1d(120)

        self.fc2 = nn.Linear(120, 84)
        if use_bn: self.bn4 = nn.BatchNorm1d(84)

        # Dropout usually goes after the activation of the hidden fully connected layers
        self.dropout = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        if self.use_bn: x = self.bn1(x)
        x = self.pool(F.relu(x))

        # Layer 2
        x = self.conv2(x)
        if self.use_bn: x = self.bn2(x)
        x = self.pool(F.relu(x))

        # Flatten
        x = x.view(-1, 16 * 5 * 5)

        # FC 1
        x = self.fc1(x)
        if self.use_bn: x = self.bn3(x)
        x = F.relu(x)
        if self.use_dropout: x = self.dropout(x)  # Dropout applied here

        # FC 2
        x = self.fc2(x)
        if self.use_bn: x = self.bn4(x)
        x = F.relu(x)
        if self.use_dropout: x = self.dropout(x)  # And/Or here

        # Output
        x = self.fc3(x)
        return x




model = LeNet5(use_dropout=False, use_bn=False)