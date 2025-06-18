
# Imports
# Import torch, torchvision, nn, optim, transforms, DataLoader, and any other necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


# File Paths
# Define the paths to the training and validation image folders
train_blue_cube = r"C:/Users/Aishik C/Desktop/AERIS/AERIS/src/DATA FOLDER/raw/images/cubes_dset/train/blue_cube"
train_no_blue_cube = r"C:/Users/Aishik C/Desktop/AERIS/AERIS/src/DATA FOLDER/raw/images/cubes_dset/train/no_cube"

val_blue_cube = r"C:/Users/Aishik C/Desktop/AERIS/AERIS/src/DATA FOLDER/raw/images/cubes_dset/val/blue_cube"
val_no_blue_cube = r"C:/Users/Aishik C/Desktop/AERIS/AERIS/src/DATA FOLDER/raw/images/cubes_dset/val/no_cube"


# Image Transformations
# Define the transformations to resize, normalize (optional), and convert images to tensors
transformation = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])



# Load Datasets

# Use ImageFolder to load training and validation datasets with transformations


# Create Dataloaders

# Use DataLoader to create train_loader and val_loader with appropriate batch sizes and shuffling


# Visualize a Sample Batch (Optional)

# Get a batch of training data to visualize and verify dataset loading (optional step)


# Define CNN Model Class

# Create a class that extends nn.Module
# Define convolutional layers, activation functions, and pooling layers in __init__
# Define the forward method to connect all layers sequentially
# Add a classifier with at least one hidden linear layer and an output layer sized to num_classes


# Instantiate the Model

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
