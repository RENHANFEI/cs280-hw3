# -*- coding: utf-8 -*-

PRINT_NN_FLAG = False;

"""
Training a Classifier
=====================
Training an image classifier
Codes derived from PyTorch Tutirial
----------------------------

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
# Argument Parser

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gpu', default=False, action="store_const", const=True)
    parser.add_argument('--data-dir', default='./data')
    return parser.parse_args()
args = parse_args()

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).
# Codes derived from CS189-hw7

import torch.nn as nn
import torch.nn.functional as F

class PreActBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut

class MyResNet(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(MyResNet, self).__init__()
        self.in_channels = 64

        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            self._make_layer(64, 64, num_blocks[0], stride=1),
            self._make_layer(64, 128, num_blocks[1], stride=2),
            self._make_layer(128, 128, num_blocks[2], stride=2),
        )

        self.classifier = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        x = self.prep(x)
        x = self.layers(x)

        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)

        x_max = F.adaptive_max_pool2d(x, (1, 1))
        x_max = x_max.view(x_max.size(0), -1)

        x = torch.cat([x_avg, x_max], dim=-1)

        x = self.classifier(x)

        return x


net = MyResNet()
net.verbose = True

if PRINT_NN_FLAG:
    print(net)

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train and test the network
# ^^^^^^^^^^^^^^^^^^^^
#
# Simply loop over our data iterator, and feed the inputs to the
# network and optimize.

import numpy as np
import pandas as pd

if (torch.cuda.is_available() and args.gpu):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

results = []


for epoch in range(args.epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    total_train = 0;
    correct_train = 0
    total_val = 0;
    correct_val = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        if (torch.cuda.is_available() and args.gpu):
            labels = labels.cuda()
            inputs = inputs.cuda()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # optimize
        lr = 0.001
        optimizer.step()

        # compute training statistics
        logits = outputs.cpu().detach().numpy()
        y_pred_train = np.argmax(logits, axis=1)
        y_train = labels.cpu().detach().numpy()
        total_train += y_train.shape[0]
        correct_train += sum(y_pred_train == y_train)
        running_loss += loss.item()


        if i % 20 == 0:

            for j, data in enumerate(testloader, 0):
                # get the inputs
                inputs, labels = data
                if (torch.cuda.is_available() and args.gpu):
                    labels = labels.cuda()
                    inputs = inputs.cuda()
                # predict outputs
                outputs = net(inputs)
                logits = outputs.cpu().detach().numpy()

                # compute validation statistics
                y_pred_train = np.argmax(logits, axis=1)
                y_train = labels.cpu().detach().numpy()
                total_val += y_train.shape[0]
                correct_val += sum(y_pred_train == y_train)
                # print("Epoch:", epoch, "\tMiniBatch:", j, "\tPartial Validation Accuracy:", correct_val / total_val)
            result = {}
            result['train_accuracy'] = correct_train / total_train
            result['val_accuracy'] = correct_val / total_val
            result['num_epochs'] = args.epochs
            result['train_loss'] = running_loss
            results.append(result)

            print("Epoch:", epoch, "\tMiniBatch:", i, "\tPartial Training Accuracy:", correct_train / total_train,
                  "\tRunning Loss:", running_loss / (i + 1), "\tPartial Validation Accuracy:", correct_val / total_val)

    print("Epoch:", epoch, "\tFinal Training Accuracy:", {correct_train / total_train})
    print("Epoch:", epoch, "\tFinal Validation Accuracy:", {correct_val / total_val})


df = pd.DataFrame([result])
print(df)

results = np.array(results)
results.dump('accuracies.txt')


