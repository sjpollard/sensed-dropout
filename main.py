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

import argparse

parser = argparse.ArgumentParser(
    description="Tool to generate sparse pixel selections from computer vision datasets with pysensors.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--type", "-t",
    choices=["r", "c"],
    default="r",
    help="Determines whether pysensors uses SSPOR or SSPOC."
)
parser.add_argument(
    "--modes", "-m",
    type=int,
    help="The number of modes to select when preparing the basis."
)
parser.add_argument(
    "--sensors", "-s",
    type=int,
    help="The number of sensors to select from the original features."
)
parser.add_argument(
    "--show-basis",
    default=False,
    action='store_true',
    help="Reshapes and displays up to 100 modes of the generated basis."
)
parser.add_argument(
    "--show-sensors",
    default=False,
    action='store_true',
    help="Reshapes and displays active sensor locations."
)
parser.add_argument(
    "--print-accuracy",
    default=False,
    action='store_true',
    help="If using SSPOC for classification, checks accuracy against train and test."
)

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
    model.fit(np.reshape(X_train, (n, height * width)), y_train, quiet=True)
    
    print(f'{len(model.get_selected_sensors())} sensors selected out of {height * width}')
    return model

def fit_SSPOR(X_train, n_basis_modes, n_sensors):
    n, height, width = X_train.shape

    basis = ps.basis.SVD(n_basis_modes=n_basis_modes)

    model = SSPOR(basis=basis, n_sensors=n_sensors)
    model.fit(np.reshape(X_train, (n, height * width)), quiet=True)
    
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

def main(args):
    X_train, y_train = get_CIFAR10()

    n, height, width = X_train.shape

    if args.type == 'r': model = fit_SSPOR(X_train, args.modes, args.sensors)
    elif args.type == 'c': model = fit_SSPOC(X_train, y_train, args.modes, args.sensors, 0.0000005)

    if args.show_basis: show_basis(model, height, width)
    if args.show_sensors: show_sensors(model, height, width)

    if args.type == 'c' and args.print_accuracy:
        print_accuracies(model, X_train, y_train, n, height, width)

if __name__ == "__main__":
    main(parser.parse_args())