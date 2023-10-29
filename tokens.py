import torch
import torchvision.datasets
import torchshow as ts

import numpy as np

from sklearn.metrics import accuracy_score

import pysensors as ps
from pysensors.classification import SSPOC
from pysensors.reconstruction import SSPOR

from patchify import patchify, unpatchify

import argparse

import data

parser = argparse.ArgumentParser(
    description="Tool to generate token masks via sparse pixel selections from computer vision datasets with pysensors.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--num", "-n",
    default=-1,
    type=int,
    help="Number of values to process from the dataset."
)
parser.add_argument(
    "--download", "-d",
    default=False,
    action='store_true',
    help="Downloads the dataset from torchvision."
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
    help="Number of modes to select when preparing the basis."
)
parser.add_argument(
    "--sensors", "-s",
    type=int,
    help="Number of sensors to select from the original features."
)
parser.add_argument(
    "--patch", "-p",
    type=int,
    help="Size of the token patches to be selected."
)
parser.add_argument(
    "--tokens", "-k",
    type=int,
    help="Number of tokens to be selected."
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
parser.add_argument(
    "--show-tokens",
    default=False,
    action='store_true',
    help="Reshapes and displays active token locations."
)
parser.add_argument(
    "--output", "-o",
    default='',
    type=str,
    help="Name of saved output file."
)

def fit_SSPOC(X_train, y_train, n_basis_modes: int, n_sensors: int, l1_penalty: float):
    n, height, width = X_train.shape

    basis = ps.basis.SVD(n_basis_modes=n_basis_modes)

    model = SSPOC(basis=basis, n_sensors=n_sensors, l1_penalty=l1_penalty)
    model.fit(np.reshape(X_train, (n, height * width)), y_train, quiet=True)
    
    print(f'{len(model.get_selected_sensors())} sensors selected out of {height * width}')
    return model

def fit_SSPOR(X_train, n_basis_modes: int, n_sensors: int):
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
    dataloader = data.get_dataloader(torchvision.datasets.CIFAR10, train=False, greyscale=True)
    X_test, y_test = next(iter(dataloader))
    X_test = X_test.numpy().squeeze()
    y_test = y_test.numpy()
    y_pred = model.predict(np.reshape(X_train, (n, height * width))[:,model.selected_sensors])
    print(f'Train accuracy: {accuracy_score(y_train, y_pred) * 100}%')

    y_pred = model.predict(np.reshape(X_test, (X_test.shape[0], height * width))[:,model.selected_sensors])
    print(f'Test accuracy: {accuracy_score(y_test, y_pred) * 100}%')

def sensors_to_patches(model: SSPOC | SSPOR, patch: int, height: int, width: int):
    patch_shape = (patch, patch)

    sensors = np.zeros(height * width)
    np.put(sensors, model.get_selected_sensors(), 1)
    sensors = np.reshape(sensors, (height, width))

    patched_sensors = patchify(sensors, patch_shape, step=patch)
    return patched_sensors

def patches_to_tokens(patched_sensors: np.ndarray, k: int):
    patch_sums = np.sum(patched_sensors, axis=(2,3))

    token_mask = np.zeros((patch_sums.shape[0] * patch_sums.shape[1]), dtype=bool)
    token_mask[np.argsort(patch_sums.ravel())[:-k-1:-1]] = True
    token_mask = token_mask.reshape((patch_sums.shape))

    print(f'{int(np.sum(token_mask))} tokens chosen out of {patched_sensors.shape[0] * patched_sensors.shape[1]}')
    return token_mask

def show_tokens(patched_sensors: np.ndarray, token_mask: np.ndarray, patch: int, height: int, width: int):
    patch_shape = (patch, patch)

    tokens = np.zeros_like(patched_sensors)
    
    token_indices = np.argwhere(token_mask)

    for index in token_indices:
        tokens[index[0]][index[1]] = np.ones(patch_shape)

    tokens = unpatchify(tokens, (height, width))

    ts.show(tokens, mode='grayscale')

def save_mask(token_mask: np.ndarray, output):
    torch.save(token_mask.from_numpy(), f'token_masks/{output}.pt')

def main(args):
    X_train, y_train = data.get_dataset_as_numpy(torchvision.datasets.CIFAR10, args.num, download=args.download, greyscale=True)

    n, height, width = X_train.shape

    if args.type == 'r': model = fit_SSPOR(X_train, args.modes, args.sensors)
    elif args.type == 'c': model = fit_SSPOC(X_train, y_train, args.modes, args.sensors, 0.0000005)

    if args.show_basis: show_basis(model, height, width)
    if args.show_sensors: show_sensors(model, height, width)

    if args.type == 'c' and args.print_accuracy:
        print_accuracies(model, X_train, y_train, n, height, width)

    patched_sensors = sensors_to_patches(model, args.patch, height, width)
    token_mask = patches_to_tokens(patched_sensors, args.tokens)

    if args.show_tokens: show_tokens(patched_sensors, token_mask, args.patch, height, width)
    
    if args.output != '': save_mask(token_mask, args.output)

if __name__ == "__main__":
    main(parser.parse_args())