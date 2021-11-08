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

import os

from dicts.datasets_dict import datasets_dict
from dicts.arch_dict import arch_dict


logs_dir = 'logs'

pretrained_models_dict = {
    'imagenet': {
        'name': 'imagenet',
    },
}

pretrained_models = {
    'model_0': ('lenet', 'all_layers', 'diffranet_synthetic', 'some text'),
    'model_1': ('alexnet', 'fc', 'diffranet_synthetic', 'some text'),
    'model_2': ('alexnet', 'all_layers', 'diffranet_synthetic', 'some text'),
    'model_3': ('alexnet', 'all_layers', 'ln84_preprocessed', 'some text'),
    'model_4': ('vgg19', 'fc', 'diffranet_synthetic', 'some text'),
    'model_5': ('vgg19', 'all_layers', 'diffranet_synthetic', 'some text'),
    'model_6': ('vgg19', 'fc', 'ln84_preprocessed', 'some text')
}

for pretrained_model in pretrained_models.values():

    arch, trainable_layers, dataset, path, info = pretrained_model
    dataset_dict = datasets_dict[dataset]
    model_dict = {}

    name = f"{arch}-{trainable_layers}-{dataset}"
    model_dict['name'] = name

    model_dict['trainable_layers'] = trainable_layers
    model_dict['path'] = os.path.join(logs_dir, name, f"{name}.ckpt")
    model_dict['arch'] = arch

    for k in ['num_classes', 'classes']:
        model_dict[k] = dataset_dict[k]
        
    for k in ['arch', 'img_size', 'grayscale', 'target_layer', 'layers']:
        model_dict[k] = arch_dict[arch]
    pretrained_models_dict[name] = model_dict
