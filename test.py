import torchvision

import data
import tokens

def main():
    train_dataloader = data.get_dataloader(torchvision.datasets.CIFAR10, batch_size=128, greyscale=True, num_workers=1, distributed=False)
    X, y = next(iter(train_dataloader))
    n, height, width = X.size(0), X.size(2), X.size(3)
    X, y = X.squeeze().numpy(), y.numpy()
    basis = tokens.get_basis('Identity', 128)
    model = tokens.fit_SSPOC(X, y, basis, 256, 0.0000005)
    patched_sensors = tokens.sensors_to_patches(model, 16, height, width)
    token_mask = tokens.patches_to_tokens(patched_sensors, 8)
    tokens.save_output('test', model, height, width, 16, patched_sensors, token_mask)

if __name__ == "__main__":
    main()