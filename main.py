import torch
import torchvision.datasets
from torchvision import transforms
import torchshow as ts

import numpy as np

from sklearn.model_selection import train_test_split

import pysensors as ps
from pysensors.classification import SSPOC

from patchify import patchify, unpatchify

def get_CIFAR10(size):
    CIFAR10 = torchvision.datasets.CIFAR10('dataset/', transform=transforms.Compose([
                                                       transforms.PILToTensor(),
                                                       transforms.Grayscale()]))
    dataloader = torch.utils.data.DataLoader(CIFAR10, batch_size=size)
    X, y = next(iter(dataloader))
    return X.numpy().squeeze(), y.numpy()

def main():
    X_train, y_train = get_CIFAR10(10)

if __name__ == "__main__":
    main()