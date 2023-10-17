import torch
import torchvision.datasets
from torchvision import transforms
import torchshow as ts

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pysensors as ps
from pysensors.classification import SSPOC
from pysensors.reconstruction import SSPOR

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

def fit_SSPOC(X_train, y_train, n_basis_modes, n_sensors, l1_penalty):
    n, height, width = X_train.shape

    basis = ps.basis.SVD(n_basis_modes=n_basis_modes)

    model = SSPOC(basis=basis, n_sensors=n_sensors, l1_penalty=l1_penalty)
    model.fit(np.reshape(X_train, (n, height * width)), y_train)
    
    print(f'{len(model.get_selected_sensors())} sensors selected out of {height * width}')
    return model

def fit_SSPOR(X_train, n_basis_modes, n_sensors):
    n, height, width = X_train.shape

    basis = ps.basis.SVD(n_basis_modes=n_basis_modes)

    model = SSPOR(basis=basis, n_sensors=n_sensors)
    model.fit(np.reshape(X_train, (n, height * width)))
    
    print(f'{len(model.get_selected_sensors())} sensors selected out of {height * width}')
    return model

def show_basis(model: SSPOC | SSPOR, height: int, width: int, n: int = 100):
    modes = model.basis.matrix_representation().shape[1]
    ts.show(np.reshape(model.basis.matrix_representation().T, (modes, height, width))[:n], mode='grayscale')

def show_sensors(model: SSPOC | SSPOR, height: int, width: int):
    sensors = np.zeros(height * width)
    np.put(sensors, model.get_selected_sensors(), 1)
    ts.show(np.reshape(sensors, (height, width)), mode='grayscale')

def print_accuracies(model: SSPOC | SSPOR, X_train, y_train, n, height, width):
    X_test, y_test = get_CIFAR10(train=False)

    y_pred = model.predict(np.reshape(X_train, (n, height * width))[:,model.selected_sensors])
    print(f'Train accuracy: {accuracy_score(y_train, y_pred) * 100}%')

    y_pred = model.predict(np.reshape(X_test, (X_test.shape[0], height * width))[:,model.selected_sensors])
    print(f'Test accuracy: {accuracy_score(y_test, y_pred) * 100}%')

def main():
    X_train, y_train = get_CIFAR10()

    n, height, width = X_train.shape

    model_r = fit_SSPOR(X_train, 100, 256)
    model_c = fit_SSPOC(X_train, y_train, 100, 256, 0.0000005)

    """ show_basis(model, height, width) """
    show_sensors(model_r, height, width)
    show_sensors(model_c, height, width)
    """ print_accuracies(model, X_train, y_train, n, height, width) """




if __name__ == "__main__":
    main()