#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os.path as osp
import os
import gc

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import vgg19, alexnet
from torchvision.transforms.transforms import Grayscale
from PIL import Image


from utils.model_utils import load_trained_model
from model.lenet import LeNet

from visualization.grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

from dicts.datasets_dict import datasets_dict
from dicts.pretrained_models_dict import pretrained_models_dict
# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'

def get_device(cuda):
    """[summary]

    Args:
        cuda ([type]): [description]

    Returns:
        [type]: [description]
    """
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths, img_size, grayscale):
    """[summary]

    Args:
        image_paths ([type]): [description]
        img_size ([type]): [description]
        grayscale ([type]): [description]

    Returns:
        [type]: [description]
    """
    images = []
    raw_images = []
    raw_imgs = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image, raw_img = preprocess(image_path, img_size, grayscale)
        images.append(image)
        raw_images.append(raw_image)
        raw_imgs.append(raw_img)
    return images, raw_images, raw_imgs


def get_classtable(classes):
    """[summary]

    Args:
        classes ([type]): [description]

    Returns:
        [type]: [description]
    """
    classes_list = sorted(classes, key=classes.get) #TODO add global get classes function
    return classes_list


def preprocess(image_path, img_size, grayscale):
    """[summary]

    Args:
        image_path ([type]): [description]
        img_size ([type]): [description]
        grayscale ([type]): [description]

    Returns:
        [type]: [description]
    """
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (img_size,) * 2)
    raw_img = Image.fromarray(raw_image)

    if grayscale:
        
        image = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ]
        )(raw_img.copy())
    else:
        image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_img.copy())

    return image, raw_image, raw_img


def save_gradient(filename, gradient):
    """[summary]

    Args:
        filename ([type]): [description]
        gradient ([type]): [description]

    Returns:
        [type]: [description]
    """
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    
    gradient = np.dot(np.abs(gradient[...,:3]), [1, 1, 1]) # to grayscale, comment for lenet
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    gradient = np.stack((gradient,)*3, axis=-1)  # to grayscale

    # cv2.imwrite(filename, np.uint8(gradient))

    return np.uint8(gradient)


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    # cv2.imwrite(filename, np.uint8(gcam))
    return np.uint8(gcam)


def save_sensitivity(filename, maps, img_size):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
model_names.extend(['vgg19_pretrained'])


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


# @main.command()
# @click.option("-i", "--image-paths", type=str, multiple=True, required=True)
# # @click.option("-a", "--arch", type=click.Choice(model_names), required=True)
# @click.option("-t", "--target-layer", type=str, required=True)
# @click.option("-k", "--topk", type=int, default=3)
# @click.option("-o", "--output-dir", type=str, default="./results")
# @click.option("--cuda/--cpu", default=True)
def demo1(image_paths, model_dict, visualizations, output_dir=''):
    """
    Visualize model responses given multiple images
    """
    # TODO: Fix demo1 calls
    print(model_dict['name'])
    model_path = model_dict['path']
    arch = model_dict['arch']
    num_classes = model_dict['num_classes']
    img_size = model_dict['img_size']
    grayscale = model_dict['grayscale']
    target_layer = model_dict['target_layer']
    classes = get_classtable(model_dict['classes'])

    top_k = num_classes
    img_names = [img.rsplit('/', 1)[-1] for img in image_paths]
    result_imgs = {k:{} for k in img_names}
    topk = num_classes
    device = get_device(True)


    # Model from torchvision
    # or from diffraction models
    # model = models.__dict__[arch](pretrained=True)
    model = load_trained_model(arch, model_path, num_classes, img_size, fine_tune=True)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Images
    images, raw_images, raw_imgs = load_images(image_paths, img_size, grayscale)
    images = torch.stack(images).to(device)
    for i, name in enumerate(img_names):
        result_imgs[name]['original'] = raw_imgs[i]

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    # =========================================================================
    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)  # sorted
    if 'vanilla' in visualizations:
        print("Vanilla Backpropagation:")

        for img in img_names:
                result_imgs[img]['vanilla'] = {}
        for i in range(topk):
            bp.backward(ids=ids[:, [i]])
            gradients = bp.generate()

            # Save results as image files
            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                vanilla = save_gradient(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-vanilla-{}.png".format(j, arch.__name__, classes[ids[j, i]]),
                    ),
                    gradient=gradients[j],
                )
                result_imgs[img_names[j]]['vanilla'][i] = vanilla

        # Remove all the hook function in the "model"
        bp.remove_hook()

    # =========================================================================
    if 'deconvolution' in visualizations:
        print("Deconvolution:")

        deconv = Deconvnet(model=model)
        _ = deconv.forward(images)

        for img in img_names:
            result_imgs[img]['deconvolution'] = {}
        for i in range(topk):
            deconv.backward(ids=ids[:, [i]])
            gradients = deconv.generate()

            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                deconvolution = save_gradient(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-deconvnet-{}.png".format(j, arch.__name__, classes[ids[j, i]]),
                    ),
                    gradient=gradients[j],
                )
                result_imgs[img_names[j]]['deconvolution'][i] = deconvolution

        deconv.remove_hook()

    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    if 'gradcam' or 'guided_grad_cam' in visualizations:
        gcam = GradCAM(model=model)
        _ = gcam.forward(images)

        gbp = GuidedBackPropagation(model=model)
        _ = gbp.forward(images)

        for img in img_names:
            result_imgs[img]['gradcam'] = {}
            result_imgs[img]['guided_gradcam'] = {}
        for i in range(topk):
            # Guided Backpropagation
            gbp.backward(ids=ids[:, [i]])
            gradients = gbp.generate()

            # Grad-CAM
            gcam.backward(ids=ids[:, [i]])
            regions = gcam.generate(target_layer=target_layer)

            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                # # Guided Backpropagation
                # save_gradient(
                #     filename=osp.join(
                #         output_dir,
                #         "{}-{}-guided-{}.png".format(j, backbone.__name__, classes[ids[j, i]]),
                #     ),
                #     gradient=gradients[j],
                # )

                # Grad-CAM
                gradcam = save_gradcam(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-gradcam-{}-{}.png".format(
                            j, arch.__name__, target_layer, classes[ids[j, i]]
                        ),
                    ),
                    gcam=regions[j, 0],
                    raw_image=raw_images[j],
                )
                result_imgs[img_names[j]]['gradcam'][i] = gradcam

                # Guided Grad-CAM
                guided_gradcam = save_gradient(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-guided_gradcam-{}-{}.png".format(
                            j, arch.__name__, target_layer, classes[ids[j, i]]
                        ),
                    ),
                    gradient=torch.mul(regions, gradients)[j],
                )
                result_imgs[img_names[j]]['guided_gradcam'][i] = guided_gradcam
    # with pytorch.no_grad():
    torch.cuda.empty_cache()
    gc.collect()


    return result_imgs
