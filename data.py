import torch
import torchvision.datasets
from torchvision import transforms
from torchvision.datasets import VisionDataset

def get_dataloader(vision_dataset: VisionDataset, batch_size: int=-1, train: bool=True, download: bool=False, greyscale: bool=False):
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

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8)