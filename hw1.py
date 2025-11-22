'For this assignment we will use the Lenet5 grayscale(1 channel) network that fits for MNIST '

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST, ImageFolder
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## running the model on NVIDIA's - RTX 2070 Q edition - GPU
print("Using device:", device)

####handling the data####
class MNISTClassifier(Dataset):
    def __init__(self, data_dir, train=True, transform=None, download=True):
        self.data = FashionMNIST(root=data_dir, train=train, download=download, transform=transform) #downloads the dataset into the data field and file
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    @property
    def classes(self):
        return self.data.classes #return the classes of the data

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]) #normalize the Tensor

train_dataset = MNISTClassifier(data_dir="./data", train=True, transform=transforms.ToTensor(), download=True) #instantiation of the datasets
test_dataset = MNISTClassifier(data_dir="./data", train=False, transform=transform, download=True)

trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True) # creating batches of 32 samples
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

idx_to_label = {i: label for i, label in enumerate(train_dataset.classes)} #dictionary associated label to number

####LeNet5 Network####
class LeNet5(nn.Module):
    def __init__(self, num_classes=10, use_dropout: bool = False, use_batchnorm: bool = False):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        #NN ARCHITECTURE#
        self.conv1= nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2= nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        if use_batchnorm: #to normalize the batche before activation
            self.bn1=nn.BatchNorm2d(6)
            self.bn2=nn.BatchNorm2d(16)

        if use_dropout: #randomly zeroing activations
            self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        #First CNN Layer
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x= F.relu(x)
        x= self.pool(x)

        #Second CNN Layer
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        #flattening the samples to vector 1x256
        x = x.view(x.size(0), -1)

        #First FC Layer
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x= self.dropout(x)

        #Second FC Layer
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x= self.dropout(x)

        #Last Layer
        x = self.fc3(x)

        return x

####Training Loop####
model = LeNet5(num_classes=10, use_dropout=False, use_batchnorm=False)
model = model.to(device) ### to delete ###
LossFunc= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_losses = []
test_losses  = []
train_accuracies = []
test_accuracies  = []
for epoch in range(num_epochs):
    model.train() #putting model in training mode
    train_running_loss= 0.0
    correct = 0
    total = 0
    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = LossFunc(outputs, labels)
        loss.backward() ## gradient desenct
        optimizer.step() ## calculating the learning step
        train_running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_train_loss = train_running_loss / len(trainloader.dataset)
    epoch_train_acc = 100.0 * correct / total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)
    print(f"Epoch {epoch}: Training loss: {epoch_train_loss:.4f}." )
    print(f"Epoch {epoch}: Training Accuracy: {epoch_train_acc:.4f}." )

#### TEST ####
model.eval()
test_running_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = LossFunc(outputs, labels)

        test_running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, dim=1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

epoch_test_loss = test_running_loss / len(testloader.dataset)
epoch_test_acc  = 100.0 * test_correct / test_total

test_losses.append(epoch_test_loss)
test_accuracies.append(epoch_test_acc)

print(f"Epoch [{epoch+1}/{num_epochs}] "
        f"- Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% "
        f"- Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")

# ### PLOTTING the data ###
# print("epochs =", num_epochs)
# print("train_losses =", train_losses)
# print("test_losses =", test_losses)
# print("train_accuracies =", train_accuracies)
# print("test_accuracies =", test_accuracies)
#
# # Create a new epoch list for the test results (only includes the final epoch)
# test_epochs = [num_epochs[-1]] # This will be [2]
#
# # ----------- PLOT LOSS -----------
# plt.figure()
# plt.plot(num_epochs, train_losses, marker='o', label="Train Loss")
# plt.plot(test_epochs, test_losses, marker='o', label="Test Loss") # USE test_epochs
#
# plt.xlabel("Epoch")
# # ... (rest of the loss plot code)
# plt.show()
#
# # ----------- PLOT ACCURACY -----------
# plt.figure()
# plt.plot(num_epochs, train_accuracies, marker='o', label="Train Accuracy")
# plt.plot(test_epochs, test_accuracies, marker='o', label="Test Accuracy") # USE test_epochs
#
# plt.xlabel("Epoch")
# # ... (rest of the accuracy plot code)
# plt.show()
















