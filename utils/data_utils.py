"""
The module contains a set of functions for preprocessing the datasets.
"""
import os
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transforms(img_size, grayscale):
    """Prepare a list of functions for transforming a dataset.
    The list varies depending on image size required by the model
    and if the model expects the input to be in grayscale.

    Args:
        img_size (int): image size required by the model.
        grayscale (bool): if True, the model expects the input to be grayscale

    Returns:
        list: a list of functions for transforming the dataset.
    """
    if grayscale:
        mean = np.array([0.5])
        std = np.array([0.25])
        my_transforms = transforms.Compose([transforms.Grayscale(), transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.25, 0.25, 0.25])
        my_transforms = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

    return my_transforms


def get_dataloaders(dataset_dict, batch_size, img_size=224, grayscale=False):
    """Create dataloaders for the given dataset.

    Args:
        dataset (dict): a dict containing information necessary to process the dataset. A subsubdict of datasets dict.
        batch_size (int): the no of images loaded in a single iteration of the model,
        img_size (int, optional): the image size required by the model. Defaults to 224.
        grayscale (bool, optional): if True, the model expects the input to be grayscale. Defaults to False.

    Returns:
        list: a list of train and val dataloaders
    """
    my_transforms = get_transforms(img_size, grayscale)
    my_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dict['path'], y), my_transforms)
                    for x, y in zip(['train', 'val'], [dataset_dict['train_dir'], dataset_dict['val_dir']])}
                    
    dataloaders = {'train': DataLoader(my_datasets['train'], batch_size=batch_size,
                                                shuffle=True, num_workers=0),
                'val': DataLoader(my_datasets['val'], batch_size=batch_size,
                                                shuffle=False, num_workers=0)}

    return [dataloaders['train'], dataloaders['val']]
