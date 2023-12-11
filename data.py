import os

import torch
import torchvision
from torchvision import transforms

def get_dataloader(dataset: str, batch_size: int=128, image_size: int=None, train: bool=True, download: bool=False, greyscale: bool=False, num_workers: int=1, distributed: bool=False):
    transform_list = [transforms.ToTensor()]
    if greyscale: transform_list.append(transforms.Grayscale())
    if image_size: transform_list.append(transforms.Resize((image_size, image_size), antialias=False))
    transform = transforms.Compose(transform_list)
    if not os.path.exists(f'datasets'):
            os.makedirs(f'datasets')
    if dataset == 'CIFAR10':
        vision_dataset = torchvision.datasets.CIFAR10(f'datasets/{dataset}', train=train, transform=transform, download=download)
    elif dataset == 'OxfordIIITPet':
        vision_dataset = torchvision.datasets.OxfordIIITPet(f'datasets/{dataset}', split=('trainval' if train else 'test'), transform=transform, download=download)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(vision_dataset, shuffle=train)
    else:
        if train: sampler = torch.utils.data.RandomSampler(vision_dataset)
        else: sampler = torch.utils.data.SequentialSampler(vision_dataset)
    
    dataloader = torch.utils.data.DataLoader(
        vision_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader