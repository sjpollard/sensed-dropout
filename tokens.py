import os
import argparse
import time

import torch
import torchvision.datasets
import torchshow as ts

import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score

import pysensors as ps
from pysensors.reconstruction import SSPOR
from pysensors.classification import SSPOC

from patchify import patchify

import data

parser = argparse.ArgumentParser(
    description="Tool to generate token masks via sparse pixel selections from computer vision datasets with pysensors.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dataset",
    choices=["CIFAR10", "CIFAR100", "OxfordIIITPet"],
    default="CIFAR10",
    help="Specifies the dataset to be used."
)
parser.add_argument(
    "--batch-size", "-n",
    default=128,
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
    "--image-size",
    default=None,
    type=int,
    help="Enforces a standard image size."
)
parser.add_argument(
    "--patch", "-p",
    default=4,
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
    "--strategy",
    choices=["frequency", "ranking"],
    default="frequency",
    help="Determines the strategy used to gather tokens for the mask."
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
    "--show-tokens",
    default=False,
    action='store_true',
    help="Reshapes and displays active token locations."
)
parser.add_argument(
    "--score",
    default=False,
    action='store_true',
    help="Calculates reconstruction error/classification accuracy on a train and test set."
)
parser.add_argument(
    "--benchmark",
    default=False,
    action='store_true',
    help="Aggregates accuracy/error for sensor range (1, 2, ... , sensors)."
)
parser.add_argument(
    "--output", "-o",
    default=False,
    action='store_true',
    help="Saves outputs to file."
)

def benchmark(args):
    dataloader = data.get_dataloader(dataset=args.dataset, batch_size=args.batch_size, image_size=args.image_size, train=True, download=args.download)
    batch = next(iter(dataloader))
    X_train, y_train = batch[0], batch[1]

    dataloader = data.get_dataloader(dataset=args.dataset, batch_size=args.batch_size, image_size=args.image_size, train=False, download=args.download)
    batch = next(iter(dataloader))
    X_test, y_test = batch[0], batch[1]

    n, c, h, w = X_train.size()

    train_scores = []
    test_scores = []

    sensors_list = range(1, args.sensors + 1)

    for sensors in sensors_list:

        model = get_model(args.fit_type, args.basis, args.modes, sensors, args.l1_penalty)

        if args.fit_type == 'r': 
            model.fit(process_tensor(X_train), quiet=True)
            train_scores.append(-model.score(process_tensor(X_train)))
            test_scores.append(-model.score(process_tensor(X_test)))
        elif args.fit_type == 'c': 
            model.fit(process_tensor(X_train), y_train.numpy(), quiet=True)
            y_pred = model.predict(process_tensor(X_train)[:,model.selected_sensors])
            train_scores.append(accuracy_score(y_train, y_pred))
            y_pred = model.predict(process_tensor(X_test)[:,model.selected_sensors])
            test_scores.append(accuracy_score(y_test, y_pred))

    if not os.path.exists('out'):
            os.makedirs('out')

    df = pd.DataFrame(data={'sensors': sensors_list, 'train': train_scores, 'test': test_scores})
    filename = f'n_{n}_t_{args.fit_type}_m_{args.modes}_s_{args.sensors}'
    df.to_csv(f'out/{filename}.csv', index=False)

def generate_tokens(args):
    dataloader = data.get_dataloader(dataset=args.dataset, batch_size=args.batch_size, image_size=args.image_size, train=True, download=args.download)
    batch = next(iter(dataloader))
    X_train, y_train = batch[0], batch[1]

    dataloader = data.get_dataloader(dataset=args.dataset, batch_size=args.batch_size, image_size=args.image_size, train=False, download=args.download)
    batch = next(iter(dataloader))
    X_test, y_test = batch[0], batch[1]

    n, c, h, w = X_train.size()

    mask_dim = (h // args.patch, w // args.patch)
    
    model = get_model(args.fit_type, args.basis, args.modes, args.sensors, args.l1_penalty)

    token_mask = fit_mask(model, args.fit_type, X_train, y_train, args.patch, args.tokens, args.strategy)

    if args.show_basis: show_basis(model, h, w)
    if args.show_sensors: show_sensors(model, h, w)
    if args.show_tokens: show_tokens(token_mask, mask_dim)

    if args.score:
        if args.fit_type == 'r':
            print(f'Train score: {-model.score(process_tensor(X_train))}')
            print(f'Test score: {-model.score(process_tensor(X_test))}')
        elif args.fit_type == 'c':
            y_pred = model.predict(process_tensor(X_train)[:,model.selected_sensors])
            print(f'Train accuracy: {accuracy_score(y_train.numpy(), y_pred) * 100}%')
            y_pred = model.predict(process_tensor(X_test)[:,model.selected_sensors])
            print(f'Test accuracy: {accuracy_score(y_test.numpy(), y_pred) * 100}%')
    
    if args.output:
        filename = f'n_{n}_t_{args.fit_type}_m_{args.modes}_s_{args.sensors}_p_{args.patch}_k_{args.tokens}_{args.strategy}'
        save_output(filename, model, h, w, token_mask, mask_dim)

def get_basis(basis: str, n_basis_modes: int):
    if basis == 'SVD':
        return ps.basis.SVD(n_basis_modes=n_basis_modes)
    elif basis == 'RandomProjection':
        return ps.basis.RandomProjection(n_basis_modes=n_basis_modes)
    elif basis == 'Identity':
        return ps.basis.Identity(n_basis_modes=n_basis_modes)
    else: return None

def get_model(fit_type: str, basis: str, modes: int, sensors: int, l1_penalty: float):
    basis = get_basis(basis, modes)
    if fit_type == 'r': model = SSPOR(basis=basis, n_sensors=sensors)
    elif fit_type == 'c': model = SSPOC(basis=basis, n_sensors=sensors, l1_penalty=l1_penalty)
    return model

def fit_mask(model: SSPOR | SSPOC, fit_type: str, x: torch.Tensor, y: torch.Tensor, patch: int, tokens: int, strategy: str):
    n, c, h, w = x.size()

    if fit_type == 'r': model.fit(process_tensor(x), quiet=True)
    elif fit_type == 'c': model.fit(process_tensor(x), y.numpy(), quiet=True)

    return mask_from_sensors(model.selected_sensors, patch, h, w, tokens, strategy)

def process_tensor(x: torch.Tensor):
    n = x.size(0)
    return (x.sum(dim=1) / 3).squeeze().reshape((n, -1)).numpy()

def mask_from_sensors(selected_sensors: list[int], patch: int, h: int, w: int, k: int, strategy: str='frequency'):    
    patch_shape = (patch, patch)

    sensors = np.zeros(h * w)
    np.put(sensors, selected_sensors, 1)
    sensors = sensors.reshape((h, w))

    patched_sensors = patchify(sensors, patch_shape, step=patch)

    mask_h, mask_w = patched_sensors.shape[:2]
    token_mask = np.zeros((mask_h * mask_w), dtype=bool)

    diff = 0
    if strategy == 'frequency':
        patch_sums = np.sum(patched_sensors, axis=(2,3))
        diff = int(k - np.sum(patch_sums))
        non_zero_indices = np.nonzero(patch_sums.ravel())[0]
        token_indices = non_zero_indices[np.argsort(patch_sums.ravel()[non_zero_indices])][:-k-1:-1]
    elif strategy == 'ranking':
        y_indices = selected_sensors // (patch * w)
        x_indices = ((selected_sensors % (patch * w)) % w) // patch
        patch_indices = y_indices * mask_w + x_indices
        unique_indices = pd.unique(patch_indices)
        diff = k - len(unique_indices)
        token_indices = unique_indices[:k]
    token_mask[token_indices] = True
    if diff > 0:
            zeros = np.argwhere(token_mask == 0).squeeze()
            new_ones = zeros[np.random.permutation(len(zeros))][:diff]
            token_mask[new_ones] = True
    return torch.from_numpy(token_mask)

def show_basis(model: SSPOR | SSPOC, h: int, w: int, n_modes: int = 100):
    modes = model.basis.matrix_representation().shape[1]
    ts.show(np.reshape(model.basis.matrix_representation().T, (modes, h, w))[:n_modes], mode='grayscale')

def show_sensors(model: SSPOR | SSPOC, h: int, w: int):
    n_sensors = len(model.selected_sensors)
    sensors = np.zeros(h * w)
    values = np.ones(n_sensors) - (np.arange(n_sensors) * 1/n_sensors)
    np.put(sensors, model.get_selected_sensors(), values)
    ts.show(sensors.reshape((h, w)), mode='grayscale')

def show_tokens(token_mask: torch.Tensor, mask_dim: tuple):
    ts.show(token_mask.reshape(mask_dim), mode='grayscale')

def random_mask(h: int, w: int, k: int):
    n = h * w
    token_mask = torch.reshape(torch.repeat_interleave(torch.tensor([False, True]), torch.tensor([n-k, k]))[torch.randperm(n)], (h, w))
    return token_mask

def save_output(filename, model : SSPOR | SSPOC, h: int, w: int, token_mask: torch.Tensor, mask_dim: tuple):
    output_modes = 100
    if not os.path.exists('token_masks'):
            os.makedirs('token_masks')
    
    modes = model.basis.matrix_representation().shape[1]
    ts.save(model.basis.matrix_representation().T.reshape((modes, h, w))[:output_modes], f'token_masks/{filename}/modes_{filename}.jpg', mode='grayscale')

    n_sensors = len(model.selected_sensors)
    sensors = np.zeros(h * w)
    values = np.ones(n_sensors) - (np.arange(n_sensors) * 1/n_sensors)
    np.put(sensors, model.get_selected_sensors(), values)
    ts.save(sensors.reshape((h, w)), f'token_masks/{filename}/sensors_{filename}.jpg', mode='grayscale')

    ts.save(token_mask.reshape(mask_dim), f'token_masks/{filename}/tokens_{filename}.jpg', mode='grayscale')

    torch.save(token_mask, f'token_masks/{filename}/token_mask_{filename}.pt')

def main(args):
    if args.benchmark: benchmark(args)
    else: generate_tokens(args)

if __name__ == "__main__":
    main(parser.parse_args())