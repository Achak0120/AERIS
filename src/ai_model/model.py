import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.datasets import ImageFolder

#Load image files
train_path = r'C:\Users\achak\OneDrive\Desktop\rover\AERIS\src\DATA FOLDER\raw\images\cubes_dset\train'
val_path = r'C:\Users\achak\OneDrive\Desktop\rover\AERIS\src\DATA FOLDER\raw\images\cubes_dset\val'
traindata = ImageFolder(train_path)
valdata = ImageFolder(val_path)


print("Classes: ", traindata.classes)
print("Classes: ", valdata.classes)

