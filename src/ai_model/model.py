import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
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
train_loader = DataLoader(traindata, batch_size=4, shuffle=True)
val_loader = DataLoader(valdata, batch_size=4, shuffle=False)

#Tester plot
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# collect batch
dataiter = iter(train_loader)
images, labels = next(dataiter)

# shows first image in mybatch
imshow(images[0])
print("Label: ", traindata.classes[labels[0]])



