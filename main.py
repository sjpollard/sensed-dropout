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

    n, height, width = X_train.shape

    n_basis_modes = 10
    l1_penalty = 0.1

    basis = ps.basis.SVD(n_basis_modes=n_basis_modes)

    model = SSPOC(basis=basis, l1_penalty=l1_penalty)
    model.fit(np.reshape(X_train, (n, height * width)), y_train)
    
    print(model.get_selected_sensors())

if __name__ == "__main__":
    main()