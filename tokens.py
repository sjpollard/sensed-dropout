import os
import argparse
import time

import torch
import torchvision.datasets
import torchshow as ts

import numpy as np

from sklearn.metrics import accuracy_score

import pysensors as ps
from pysensors.classification import SSPOC
from pysensors.reconstruction import SSPOR

from patchify import patchify, unpatchify

import data

parser = argparse.ArgumentParser(
    description="Tool to generate token masks via sparse pixel selections from computer vision datasets with pysensors.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--batch-size", "-n",
    default=-1,
    type=int,
    help="Number of values to process from the dataset for each fit."
)
parser.add_argument(
    "--download", "-d",
    default=False,
    action='store_true',
    help="Downloads the dataset from torchvision."
)
parser.add_argument(
    "--fit-type", "-t",
    choices=["r", "c"],
    default="r",
    help="Determines whether pysensors uses SSPOR or SSPOC."
)
parser.add_argument(
    "--basis", "-b",
    choices=["SVD", "RandomProjection", "Identity"],
    default="Identity",
    help="Determines the basis to use."
)
parser.add_argument(
    "--modes", "-m",
    default=1,
    type=int,
    help="Number of modes to select when preparing the basis."
)
parser.add_argument(
    "--sensors", "-s",
    default=1,
    type=int,
    help="Number of sensors to select from the original features."
)
parser.add_argument(
    "--l1-penalty", "-l",
    default=0.001,
    type=float,
    help="Strength of L1 regularisation."
)
parser.add_argument(
    "--patch", "-p",
    default=0,
    type=int,
    help="Size of the token patches to be selected."
)
parser.add_argument(
    "--tokens", "-k",
    default=0,
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
    "--print-error",
    default=False,
    action='store_true',
    help="If using SSPOR for reconstruction, checks error against train and test."
)
parser.add_argument(
    "--show-tokens",
    default=False,
    action='store_true',
    help="Reshapes and displays active token locations."
)
parser.add_argument(
    "--output", "-o",
    default=False,
    action='store_true',
    help="Saves outputs to file."
)

def get_basis(basis: str, n_basis_modes: int):
    if basis == 'SVD':
        return ps.basis.SVD(n_basis_modes=n_basis_modes)
    elif basis == 'RandomProjection':
        return ps.basis.RandomProjection(n_basis_modes=n_basis_modes)
    elif basis == 'Identity':
        return ps.basis.Identity(n_basis_modes=n_basis_modes)
    else: return None

def show_basis(model: SSPOC | SSPOR, h: int, w: int, n_modes: int = 100):
    modes = model.basis.matrix_representation().shape[1]
    ts.show(np.reshape(model.basis.matrix_representation().T, (modes, h, w))[:n_modes], mode='grayscale')

def show_sensors(model: SSPOC | SSPOR, h: int, w: int):
    sensors = np.zeros(h * w)
    np.put(sensors, model.get_selected_sensors(), 1)
    ts.show(np.reshape(sensors, (h, w)), mode='grayscale')

def print_classification_accuracy(model: SSPOC, X_train, y_train, batch_size: int):
    n, h, w = X_train.shape
    dataloader = data.get_dataloader(torchvision.datasets.CIFAR10, batch_size=batch_size, train=False, greyscale=True)
    batch = next(iter(dataloader))
    X_test, y_test = batch[0].squeeze().numpy(), batch[1].numpy()

    y_pred = model.predict(X_train.reshape((n, -1))[:,model.selected_sensors])
    print(f'Train accuracy: {accuracy_score(y_train, y_pred) * 100}%')

    y_pred = model.predict(X_test.reshape((batch_size, -1))[:,model.selected_sensors])
    print(f'Test accuracy: {accuracy_score(y_test, y_pred) * 100}%')

def print_reconstruction_error(model: SSPOR, X_train, batch_size: int):
    n, h, w = X_train.shape
    dataloader = data.get_dataloader(torchvision.datasets.CIFAR10, batch_size=batch_size, train=False, greyscale=True)
    batch = next(iter(dataloader))
    X_test = batch[0].squeeze().numpy()

    print(f'Train score: {-model.score(X_train.reshape((n, -1)))}')

    print(f'Test score: {-model.score(X_test.reshape((n, -1)))}')

def sensors_to_patches(model: SSPOC | SSPOR, patch: int, h: int, w: int):
    patch_shape = (patch, patch)

    sensors = np.zeros(h * w)
    np.put(sensors, model.get_selected_sensors(), 1)
    sensors = sensors.reshape((h, w))

    patched_sensors = patchify(sensors, patch_shape, step=patch)
    return patched_sensors

def patches_to_tokens(patched_sensors: np.ndarray, k: int):
    patch_sums = np.sum(patched_sensors, axis=(2,3))

    token_mask = np.zeros((patch_sums.shape[0] * patch_sums.shape[1]), dtype=bool)
    token_mask[np.argsort(patch_sums.ravel())[:-k-1:-1]] = True
    token_mask = token_mask.reshape((patch_sums.shape))
    return token_mask

def random_mask(h: int, w: int, k: int):
    n = h * w
    token_mask = torch.reshape(torch.repeat_interleave(torch.tensor([False, True]), torch.tensor([n-k, k]))[torch.randperm(n)], (h, w))
    return token_mask

def show_tokens(patched_sensors: np.ndarray, token_mask: np.ndarray, patch: int, h: int, w: int):
    patch_shape = (patch, patch)

    tokens = np.zeros_like(patched_sensors)
    
    token_indices = np.argwhere(token_mask)

    for index in token_indices:
        tokens[index[0]][index[1]] = np.ones(patch_shape)

    tokens = unpatchify(tokens, (h, w))

    ts.show(tokens, mode='grayscale')

def save_output(filename, model : SSPOC | SSPOR, h: int, w: int, token_mask: np.ndarray):
    output_modes = 100
    if not os.path.exists(f'token_masks'):
            os.makedirs(f'token_masks')
    
    modes = model.basis.matrix_representation().shape[1]
    ts.save(model.basis.matrix_representation().T.reshape((modes, h, w))[:output_modes], f'token_masks/{filename}/modes_{filename}.jpg', mode='grayscale')

    sensors = np.zeros(h * w)
    np.put(sensors, model.get_selected_sensors(), 1)
    ts.save(sensors.reshape((h, w)), f'token_masks/{filename}/sensors_{filename}.jpg', mode='grayscale')

    ts.save(token_mask, f'token_masks/{filename}/tokens_{filename}.jpg', mode='grayscale')

    torch.save(torch.from_numpy(token_mask), f'token_masks/{filename}/token_mask_{filename}.pt')

def get_model(fit_type: str, basis: str, modes: int, sensors: int, l1_penalty: float):
    basis = get_basis(basis, modes)
    if fit_type == 'r': model = SSPOR(basis=basis, n_sensors=sensors)
    elif fit_type == 'c': model = SSPOC(basis=basis, n_sensors=sensors, l1_penalty=l1_penalty)
    return model

def main(args):
    dataloader = data.get_dataloader(torchvision.datasets.CIFAR10, batch_size=args.batch_size, download=args.download, greyscale=True)
    batch = next(iter(dataloader))
    X_train, y_train = batch[0].squeeze().numpy(), batch[1].numpy()

    n, h, w = X_train.shape

    model = get_model(args.fit_type, args.basis, args.modes, args.sensors, args.l1_penalty)

    start_time = time.time()
    if args.fit_type == 'r': model.fit(X_train.reshape((n, -1)))
    elif args.fit_type == 'c': model.fit(X_train.reshape((n, -1)), y_train)
    print(time.time() - start_time)

    if args.show_basis: show_basis(model, h)
    if args.show_sensors: show_sensors(model, h, w)

    if args.print_accuracy: print_classification_accuracy(model, X_train, y_train, args.batch_size)
    elif args.print_error: print_reconstruction_error(model, X_train, args.batch_size)

    if args.patch != 0: patched_sensors = sensors_to_patches(model, args.patch, h, w)
    if args.tokens != 0: token_mask = patches_to_tokens(patched_sensors, args.tokens)

    if args.show_tokens: show_tokens(patched_sensors, token_mask, args.patch, h, w)
    
    if args.output:
        filename = f'n_{n}_t_{args.fit_type}_m_{args.modes}_s_{args.sensors}_p_{args.patch}_k_{args.tokens}'
        save_output(filename, model, h, w, token_mask)

if __name__ == "__main__":
    main(parser.parse_args())