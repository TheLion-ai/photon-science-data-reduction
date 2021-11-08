""" A configuration file for model training.
"""
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam, SGD, lr_scheduler
from torchvision.models import alexnet, resnet50, vgg19
from torchvision.transforms.transforms import Grayscale

from dicts.datasets_dict import datasets_dict
from dicts.pretrained_models_dict import pretrained_models_dict

from model.train import train
from model.lenet import LeNet

root_dir = 'home/bk/sfx/model'
tune_fc_only = True
dataset = datasets_dict['ln84_preprocessed']
arch = vgg19
encoder = 'vgg19_all_layers'
img_size = 224
loss = BCELoss if dataset['multilabel'] else CrossEntropyLoss
grayscale = True if arch.__name__ == 'LeNet' else False

config = {
    'dataloader': {
        'dataset_dict': dataset,
        'batch_size': 1024,
        'img_size': img_size,
        'grayscale': grayscale
        },

    'model': {
            'arch': arch,
            # 'encoder_path': pretrained_models_dict[encoder]['path'],
            'num_classes': dataset['num_classes'],
            'img_size': img_size,
            'tune_fc_only': tune_fc_only,
            'optimizer': Adam,
            'loss': loss,
            'lr': 1e-3
        },

    'trainer': {
        'gpus': [0, 1, 2],
        'max_epochs': 3000
    },

    'model_id': f"2910_notf_{img_size}_{arch.__name__}_tune_fc_only{tune_fc_only}_{dataset['name']}"
}

if __name__ == '__main__':

    dataloader_params = config['dataloader']
    model_params = config['model']
    trainer_params = config['trainer']
    model_id = config['model_id']
    train(dataloader_params, model_params, trainer_params, model_id)

