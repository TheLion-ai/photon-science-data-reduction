from torchvision.models import alexnet, resnet50, vgg19
from model.lenet import LeNet

from dicts.layers_dict import layers_dict


arch_dict = {
    'lenet': {
        'arch': LeNet,
        'img_size': 224,
        'grayscale': False,
        'target_layer': 'features.7',
        'layers': layers_dict['lenet']
    },
    'alexnet': {
        'arch': alexnet,
        'img_size': 224,
        'grayscale': False,
        'target_layer': 'avgpool',
        'layers': layers_dict['alexnet']
    },
    'vgg19': {
        'arch': vgg19,
        'img_size': 224,
        'grayscale': False,
        'target_layer': 'avgpool',
        'layers': layers_dict['vgg19'],
    }
}