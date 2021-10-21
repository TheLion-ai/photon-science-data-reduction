"""
This file contains all information related to the available pretrained models.
pretrained_models_dict keys:
:key (str): model name,
subkeys:
:name (str): the model name,
:path (str): the relative path to the pretrained model,
:backbone (class): class with the architecture that was used to build the model,
:num_classes (int): total number of classes that the model classifies,
:classes (dict): a dict with class names as keys and class indices as values,
:img_size (int): input image size,
:grayscale (bool): the input image is in grayscale if true, else in rgb,
:target_layer (str): name of the layer used by Grad-CAM, Guided Grad-CAM, Vanilla Grad-CAM and deconvolution for visualization.
num of classes and ordered class names
- input parameters: img_size and if input is in grayscale
- target layer for visualizations
"""

from torchvision.models import alexnet, resnet50, vgg19
from model.lenet import LeNet


pretrained_models_dict = {
    'imagenet': {
        'name': 'imagenet',
    },
    'alexnet_all_layers': {
        'name': 'alexnet_all_layers',
        'path': '/home/bk/sfx/model/logs/alexnet_full_tune_imagenet-epoch=12-val_loss=0.18.ckpt',
        'backbone': alexnet,
        'num_classes': 5,
        'classes': {
                'blank': 0,
                'no-crystal': 1,
                'weak': 2,
                'good': 3,
                'strong': 4
        },
        'img_size': 224,
        'grayscale': False,
        'target_layer': 'avgpool'
    },
    'alexnet_fc': {
        'name': 'alexnet_fc',
        'path': '/home/bk/sfx/model/logs/alexnet_fine_tune_imagenet-epoch=22-val_loss=0.09.ckpt',
        'backbone': alexnet,
        'num_classes': 5,
        'classes': {
                'blank': 0,
                'no-crystal': 1,
                'weak': 2,
                'good': 3,
                'strong': 4
        },
        'img_size': 224,
        'grayscale': False,
        'target_layer': 'avgpool'
    },
    'alexenet_diffranet': {
        'name': 'alexnet_diffranet',
        'path': '/home/bk/sfx/model/logs/alexnet_fine_tune_diffranet-epoch=22-val_loss=0.22.ckpt',
        'backbone': alexnet,
        'num_classes': 2,
        'classes': {
            'no_diffraction': 0,
            'diffraction': 1
        },
        'img_size': 224,
        'grayscale': False,
        'target_layer': 'avgpool'
    },
    'lenet_32': {
        'name': 'lenet_32',
        'path': "/home/bk/sfx/model/logs/32_LeNet_tune_fc_onlyFalse_diffranet_synthetic/-epoch=1497-val_loss=0.13-accuracy=0.00.tmp_end.ckpt",
        'backbone': LeNet,
        'num_classes': 5,
        'classes': {
                'blank': 0,
                'no-crystal': 1,
                'weak': 2,
                'good': 3,
                'strong': 4
        },
        'img_size': 32,
        'grayscale': True,
        'target_layer': 'features.7'
    },
    'lenet_224': {
        'name': 'lenet_224',
        'path': "/home/bk/sfx/model/logs/224_LeNet_tune_fc_onlyFalse_diffranet_synthetic/-epoch=182-val_loss=0.96-accuracy=0.00.tmp_end.ckpt",
        'backbone': LeNet,
        'num_classes': 5,
        'classes': {
                'blank': 0,
                'no-crystal': 1,
                'weak': 2,
                'good': 3,
                'strong': 4
        },
        'img_size': 224,
        'grayscale': True,
        'target_layer': 'features.7'
    },
    'vgg19_fc': {
        'name': 'vgg19_fc',
        'path': "/home/bk/sfx/model/logs/2009_1154vgg19_tune_fc_onlyTrue_diffranet_synthetic/-epoch=02-val_loss=0.09-accuracy=0.00.ckpt",
        'backbone': vgg19,
        'num_classes': 5,
        'classes': {
                'blank': 0,
                'no-crystal': 1,
                'weak': 2,
                'good': 3,
                'strong': 4
        },
        'img_size': 224,
        'grayscale': False,
        'target_layer': 'avgpool'
    },
    'vgg19_all_layers': {
        'name': 'vgg19_all_layers',
        'path': "/home/bk/sfx/model/logs/2009_1154vgg19_tune_fc_onlyFalse_diffranet_synthetic/-epoch=08-val_loss=0.09-accuracy=0.00.ckpt",
        'backbone': vgg19,
        'num_classes': 5,
        'classes': {
                'blank': 0,
                'no-crystal': 1,
                'weak': 2,
                'good': 3,
                'strong': 4
        },
        'img_size': 224,
        'grayscale': False,
        'target_layer': 'avgpool'
    }
}
