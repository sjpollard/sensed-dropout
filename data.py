import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset

def get_dataset_as_numpy(vision_dataset: VisionDataset, batch_size: int=-1, train: bool=True, download: bool=False, greyscale: bool=False, num_workers: int=1):
    if greyscale:
        transform = transforms.Compose([transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        transforms.Resize((128, 128), antialias=False)])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((128, 128), antialias=False)])
        
    dataset = vision_dataset(f'datasets/{vision_dataset.__name__}', transform=transform, train=train, download=download)
    if batch_size == -1:
        batch_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    X, y = next(iter(dataloader))
    return X.numpy().squeeze(), y.numpy()

def get_dataloader(vision_dataset: VisionDataset, batch_size: int=-1, train: bool=True, download: bool=False, greyscale: bool=False, num_workers: int=1, distributed: bool=False):
    if greyscale:
        transform = transforms.Compose([transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        transforms.Resize((128, 128), antialias=False)])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((128, 128), antialias=False)])
        
    dataset = vision_dataset(f'datasets/{vision_dataset.__name__}', transform=transform, train=train, download=download)
    if batch_size == -1:
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
        pin_memory=True
    )

    return dataloader