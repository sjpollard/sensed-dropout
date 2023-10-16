import torch
import torchvision.datasets
from torchvision import transforms
import torchshow as ts

import numpy as np

from sklearn.model_selection import train_test_split

import pysensors as ps
from pysensors.classification import SSPOC

from patchify import patchify, unpatchify

def get_CIFAR10(size=-1, train=True):
    CIFAR10 = torchvision.datasets.CIFAR10('dataset/', transform=transforms.Compose([
                                                       transforms.PILToTensor(),
                                                       transforms.Grayscale()]), train=train)
    if size == -1:
        size = len(CIFAR10)
    dataloader = torch.utils.data.DataLoader(CIFAR10, batch_size=size)
    X, y = next(iter(dataloader))
    return X.numpy().squeeze(), y.numpy()

def show_basis(model: SSPOC, height: int, width: int):
    modes = model.basis.matrix_representation().shape[1]
    ts.show(np.reshape(model.basis.matrix_representation().T, (modes, height, width)), mode='grayscale')

def show_sensors(model: SSPOC, height: int, width: int):
    sensors = np.zeros(height * width)
    np.put(sensors, model.get_selected_sensors(), 1)
    ts.show(np.reshape(sensors, (height, width)), mode='grayscale')

def main():
    X_train, y_train = get_CIFAR10(10000)

    n, height, width = X_train.shape

    n_basis_modes = 20
    l1_penalty = 0.00001

    basis = ps.basis.SVD(n_basis_modes=n_basis_modes)

    model = SSPOC(basis=basis, l1_penalty=l1_penalty)
    model.fit(np.reshape(X_train, (n, height * width)), y_train)
    
    show_basis(model, height, width)
    show_sensors(model, height, width)

if __name__ == "__main__":
    main()