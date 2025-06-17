import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

#Load image files
train_path = r'C:/Users/achak/OneDrive/Desktop/rover/AERIS/src/DATA FOLDER/raw/images/cubes_dset/train'
val_path = r'C:/Users/achak/OneDrive/Desktop/rover/AERIS/src/DATA FOLDER/raw/images/cubes_dset/val'


# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

traindata = ImageFolder(train_path, transform=transform)
valdata = ImageFolder(val_path, transform=transform)


# Dataloaders
train_loader = DataLoader(traindata, batch_size=5, shuffle=True)
val_loader = DataLoader(valdata, batch_size=5, shuffle=False)

# collect batch
dataiter = iter(train_loader)
images, labels = next(dataiter)


# Model Infrastructure
class RoverCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.maxpool3 = nn.MaxPool2d(2)

    def forward(self, x):
        pass