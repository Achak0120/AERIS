# Imports
# Import torch, torchvision, nn, optim, transforms, DataLoader, and any other necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


# File Paths
# Define the paths to the training and validation image folders
train_blue_cube = r"C:\Users\Aishik C\Desktop\AERIS\AERIS\src\DATA FOLDER\raw\images\cubes_dset\train\blue_cube"
train_no_blue_cube = r"C:\Users\Aishik C\Desktop\AERIS\AERIS\src\DATA FOLDER\raw\images\cubes_dset\train\no_cube"

val_blue_cube = r"C:\Users\Aishik C\Desktop\AERIS\AERIS\src\DATA FOLDER\raw\images\cubes_dset\val\blue_cube"
val_no_blue_cube = r"C:\Users\Aishik C\Desktop\AERIS\AERIS\src\DATA FOLDER\raw\images\cubes_dset\val\no_cube"

train_path = r"C:\Users\Aishik C\Desktop\AERIS\AERIS\src\DATA FOLDER\raw\images\cubes_dset\train"
val_path = r"C:\Users\Aishik C\Desktop\AERIS\AERIS\src\DATA FOLDER\raw\images\cubes_dset\val"

# Image Transformations
# Define the transformations to resize, normalize (optional), and convert images to tensors
transformation = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])


# Load Datasets - Training & Validation
# Use ImageFolder to load training and validation datasets with transformations
train_dataset = datasets.ImageFolder(
    train_path,
    transform=transformation
)

val_dataset = datasets.ImageFolder(
    val_path,
    transform=transformation
)

# Create Dataloaders
# Use DataLoader to create train_loader and val_loader with appropriate batch sizes and shuffling
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)
'''
Visualize a Sample Batch (Optional) --> Skipped
Get a batch of training data to visualize and verify dataset loading (optional step) --> Skipped
'''

# Define the GPU as a device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Define CNN Model Class
class DetectCNN(nn.Module):
    def __init__(self) -> None:
        super(DetectCNN, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.normal1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.5)
        
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.normal2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=200704, out_features=1)
# Create a class that extends nn.Module
# Define convolutional layers, activation functions, and pooling layers in __init__
# Define the forward method to connect all layers sequentially
# Add a classifier with at least one hidden linear layer and an output layer sized to num_classes
    def forward(self, t):
        t = self.conv1(t)
        t = self.normal1(t)
        t = self.relu1(t)
        t = self.pool1(t)
        t = self.dropout1(t)
        
        t = self.conv2(t)
        t = self.normal2(t)
        t = self.relu2(t)
        t = self.pool2(t)
        t = self.dropout2(t)
        
        t = self.flatten(t)
        t = self.linear(t)
        return t


# Instantiate the Model
model = DetectCNN().to(device)
print(model)
# Create an instance of the model and pass the number of output classes


# Define Loss Function and Optimizer

# Choose a suitable loss function like CrossEntropyLoss
# Choose an optimizer like Adam or SGD


# Training Loop

# Loop over multiple epochs
# For each epoch, loop through train_loader
# In each batch: zero gradients, forward pass, compute loss, backpropagate, optimizer step
# Optionally print training progress and loss


# Validation Loop

# After each epoch (or at the end), run model in evaluation mode
# Loop through val_loader without gradient calculation
# Compute accuracy or other metrics


# Save Trained Model (Optional)

# Save the model state_dict for later use


# Load Model (Optional)

# Load the saved model weights if needed in a separate script
