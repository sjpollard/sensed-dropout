# Sparse Tokens

Experimenting with non-adaptive selection of image tokens for vision transformers via data driven sparse sensor selection.

## Installation

Clone sparse-tokens

```
git clone https://github.com/sjpollard/sparse-tokens.git
```

Prepare anaconda environment, `mamba` also works

```
conda env create -f environment.yml
```

## Run

Selecting sensors with SSPOR

```
python tokens.py --type r --modes m --sensors s --patch p
```

Selecting sensors with SSPOC

```
python tokens.py --type c --modes m --sensors s --patch p
```

Displaying basis, sensors and tokens

```
python tokens.py ... --show-basis --show-sensors --show-tokens
```

## Arguments

### main.py

- `--num, -n` (int): Number of values to process from the dataset.
- `--download, -d` (on/off): Downloads the dataset from torchvision.
- `--type, -t` (str): `'r'` or `'c'` depending on whether a SSPOR or SSPOC model should be used
- `--modes, -m` (int): Number of modes to select when preparing the basis
- `--sensors, -s` (int): Number of sensors to select from the original features
- `--patch, -p` (int): Size of the token patches to be selected
- `--tokens, -k` (int): Number of tokens to be selected
- `--show-basis` (on/off): Reshapes and displays up to 100 modes of the generated basis
- `--show-sensors` (on/off): Reshapes and displays active sensor locations
- `--print-accuracy` (on/off): If using SSPOC for classification, checks accuracy against train and test
- `--show-tokens` (on/off): Reshapes and displays active token locations

## Acknowledgements
This project borrows ideas and/or code from the following preceding works:

```
@article{de Silva2021,
  doi = {10.21105/joss.02828},
  url = {https://doi.org/10.21105/joss.02828},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {58},
  pages = {2828},
  author = {Brian M. de Silva and Krithika Manohar and Emily Clark and Bingni W. Brunton and J. Nathan Kutz and Steven L. Brunton},
  title = {PySensors: A Python package for sparse sensor placement},
  journal = {Journal of Open Source Software}
}
```