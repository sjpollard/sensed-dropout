import torch
import torchvision.datasets
from torchvision import transforms

def get_dataloader(vision_dataset: torchvision.datasets.VisionDataset, batch_size: int=-1, train: bool=True, download: bool=False):
    dataset = vision_dataset(f'datasets/{vision_dataset.__name__}', transform=transforms.Compose([
                                                       transforms.PILToTensor(),
                                                       transforms.Grayscale(),
                                                       transforms.Resize((128, 128), antialias=False)]),
                                                       train=train,
                                                       download=download)
    if batch_size == -1:
        batch_size = len(dataset)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8)