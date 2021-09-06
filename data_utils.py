import os
from sys import prefix
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transforms():
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])
    my_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

    return my_transforms


def get_dataloaders(dataset, batch_size):
    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset['path'], y), get_transforms())
                    for x, y in zip(['train', 'val'], [dataset['train_dir'], dataset['val_dir']])}
                    
    dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=batch_size,
                                                shuffle=True, num_workers=0),
                'val': DataLoader(image_datasets['val'], batch_size=batch_size,
                                                shuffle=False, num_workers=0)}

    return [dataloaders['train'], dataloaders['val']]
