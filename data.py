import os

import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset

def get_dataset_as_numpy(vision_dataset: VisionDataset, batch_size: int=-1, train: bool=True, download: bool=False, greyscale: bool=False, num_workers: int=1):
    transform_list = []
    if greyscale: transform_list.append(transforms.Grayscale())
    transform_list.extend([transforms.ToTensor(),
                          transforms.Resize((128, 128), antialias=False)])
    transform = transforms.Compose(transform_list)
    if not os.path.exists(f'datasets'):
            os.makedirs(f'datasets')    
    dataset = vision_dataset(f'datasets/{vision_dataset.__name__}', transform=transform, train=train, download=download)
    if batch_size == -1:
        batch_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    X, y = next(iter(dataloader))
    return X.numpy().squeeze(), y.numpy()

def get_dataloader(vision_dataset: VisionDataset, batch_size: int=None, image_size: int=None, train: bool=True, download: bool=False, greyscale: bool=False, num_workers: int=1, distributed: bool=False):
    transform_list = [transforms.ToTensor()]
    if greyscale: transform_list.append(transforms.Grayscale())
    if image_size: transform_list.append(transforms.Resize((image_size, image_size), antialias=False))
    transform = transforms.Compose(transform_list)
    if not os.path.exists(f'datasets'):
            os.makedirs(f'datasets')
    dataset = vision_dataset(f'datasets/{vision_dataset.__name__}', transform=transform, train=train, download=download)
    if batch_size == None:
        batch_size = len(dataset)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=train)
    else:
        if train: sampler = torch.utils.data.RandomSampler(dataset)
        else: sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader