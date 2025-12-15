import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import torch.nn.functional as F

import matplotlib.pyplot as plt

##### DEVICE - NVIDIA RTX 2070 Q EDITION #####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

##### DATASET WRAPPER ####
class MNISTClassifier(Dataset):
    def __init__(self, data_dir, train=True, transform=None, download=True):
        self.data = FashionMNIST(
            root=data_dir,
            train=train,
            download=download,
            transform=transform
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

####DATA LOADING ####
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MNISTClassifier(data_dir="data", train=True, transform=transform, download=True)
test_dataset  = MNISTClassifier(data_dir="data", train=False, transform=transform, download=True)

batch_size = 128
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

idx_to_label = {i: label for i, label in enumerate(train_dataset.classes)}

#### MODEL DEFINITION ####
class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, use_dropout: bool = False, use_batchnorm: bool = False, use_weight_decay: bool = False):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.use_weight_decay = use_weight_decay

        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, 3)    # 1x28x28 -> 32x26x26
        self.pool  = nn.MaxPool2d(2, 2)     # -> 32x13x13
        self.conv2 = nn.Conv2d(32, 64, 3)   # -> 64x11x11, then pool -> 64x5x5

        # Fully-connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 84)   # 1600 -> 84
        self.fc2 = nn.Linear(84, 64)           # 84 -> 64
        self.fc3 = nn.Linear(64, num_classes)  # 64 -> 10

        # BatchNorm layers (match channels!)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)

        # Dropout
        if use_dropout:
            self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        # Conv 1
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv 2
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten: (batch, 64*5*5 = 1600)
        x = x.view(x.size(0), -1)

        # FC 1
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)  # only after fc1

        # FC 2
        x = F.relu(self.fc2(x))

        # Output
        x = self.fc3(x)
        return x

#### TRAINING FUNCTION ####
def train_one_model(model, trainloader, testloader, num_epochs: int = 15, lr: float = 0.001, wd: float = 0.0):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        # ---- TRAIN ----
        model.train()
        correct = 0
        total = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100.0 * correct / total
        train_accs.append(train_acc)

        # ---- TEST ----
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100.0 * correct / total
        test_accs.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"| Train Acc: {train_acc:.2f}% "
              f"| Test Acc: {test_acc:.2f}%")

    return train_accs, test_accs

#### TRAIN & TEST 4 MODEL VARIATIONS ####
num_epochs = 25
results = {}
curves = {}

# 1. No regularization
print("\n==== Training: No Regularization ====")
model_none = LeNet5(use_dropout=False, use_batchnorm=False, use_weight_decay=False)
train_none, test_none = train_one_model(model_none, trainloader, testloader, num_epochs=num_epochs, lr=0.001, wd=0.0)
results["None"] = (train_none[-1], test_none[-1])
curves["None"] = (train_none, test_none)

# 2. Dropout
print("\n==== Training: Dropout ====")
model_do = LeNet5(use_dropout=True, use_batchnorm=False, use_weight_decay=False)
train_do, test_do = train_one_model(model_do, trainloader, testloader, num_epochs=num_epochs, lr=0.001, wd=0.0)
results["Dropout"] = (train_do[-1], test_do[-1])
curves["Dropout"] = (train_do, test_do)

# 3. Weight Decay
print("\n==== Training: Weight Decay ====")
model_wd = LeNet5(use_dropout=False, use_batchnorm=False, use_weight_decay=True)
train_wd, test_wd = train_one_model(model_wd, trainloader, testloader, num_epochs=num_epochs, lr=0.001, wd=1e-4)
results["Weight Decay"] = (train_wd[-1], test_wd[-1])
curves["Weight Decay"] = (train_wd, test_wd)

# 4. BatchNorm
print("\n==== Training: BatchNorm ====")
model_bn = LeNet5(use_dropout=False, use_batchnorm=True, use_weight_decay=False)
train_bn, test_bn = train_one_model(model_bn, trainloader, testloader, num_epochs=num_epochs, lr=0.001, wd=0.0)
results["BatchNorm"] = (train_bn[-1], test_bn[-1])
curves["BatchNorm"] = (train_bn, test_bn)

#### PRINT SUMMARY TABLE ####
print("\n===== FINAL RESULTS =====")
print(f"{'Model':<15} | {'Train Acc (%)':<15} | {'Test Acc (%)':<15}")
print("-" * 50)
for name, (train_acc, test_acc) in results.items():
    print(f"{name:<15} | {train_acc:<15.2f} | {test_acc:<15.2f}")

#### PLOT RESULTS ####
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(10, 5))
for name, (train_curve, test_curve) in curves.items():
    plt.plot(epochs, test_curve, marker='o', label=f"{name} Test")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy vs Epoch for Different Regularizations")
plt.legend()
plt.grid(True)
plt.show()
