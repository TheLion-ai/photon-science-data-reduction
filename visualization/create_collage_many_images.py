import os

import numpy as np
from numpy.core.numeric import full
import pandas as pd

from PIL import Image, ImageDraw, ImageFont

from utils.basic_utils import prepare_path
from visualization.grad_cam_main import demo1
from evaluation.collage_utils import *
from dicts.pretrained_models_dict import pretrained_models_dict
from dicts.datasets_dict import datasets_dict


def create_visulizations_collage_many_imgs(img_paths, vis, model_dict, dataset_dict, draw_visualization_func, caption=False):
    
    output_dir = prepare_path(model_dict['name'], dataset_dict['name'])
    classes_list = sorted(model_dict['classes'], key=model_dict['classes'].get)

    root_img_path = os.path.join('datasets', dataset_dict['name'], 'test')
    

    # TODO global img)paths in demo1
    results = {}
    full_img_paths = []
    for key in img_paths:
        full_img_paths.extend([os.path.join(root_img_path, str(key), img_path) for img_path in img_paths[key]])

    results = demo1(full_img_paths, model_dict, visualizations=[vis])
    # results = {**results, **result}

    
    x, y = (0, 0)
    page = Image.new('RGB', (5*IMG_SIZE+INTERFACING*4+PADDING*2, 5*IMG_SIZE+4*INTERFACING+LABEL_Y*2+PADDING), '#FFFFFF')

    page, x, y = draw_caption(page, (x, y), classes_list, model_dict['name'], visualization=vis)
    page, x, y = draw_vis_many_images(page, (x, y), classes_list, results, img_paths, vis)
            
    page.save(os.path.join(output_dir, f'{model_dict["name"]}_{vis}_collage_many_imgs.png'))


if __name__ == '__main__':
    csv_set = 'undecided' #'undecided' #'misclassified'
    # TODO if vgg19 clip do 4
    # TODO dla deconv i vgg19 clip do 3
    img_paths = {
        0: ['fake_19166.png', 'fake_12408.png', 'fake_12413.png', 'fake_12428.png', 'fake_12429.png'],
        1: ['fake_19635.png', 'fake_12401.png', 'fake_12402.png',  'fake_12404.png', 'fake_12410.png'],
        2: ['fake_12554.png', 'fake_12403.png', 'fake_12406.png', 'fake_12412.png', 'fake_12583.png'],
        3: ['fake_18850.png', 'fake_12405.png', 'fake_12409.png', 'fake_12411.png', 'fake_12415.png'],
        4: ['fake_13532.png', 'fake_12407.png', 'fake_12418.png', 'fake_12419.png', 'fake_12424.png']
    }

    dataset_dict = datasets_dict['diffranet']['synthetic']

    # for key in pretrained_models_dict.keys():
    #     if key in ['imagenet', 'alexnet_diffranet']:
    #         continue

    model_dict = pretrained_models_dict['alexnet_fc']
    vis = VISUALIZATIONS[1]
    create_visulizations_collage_many_imgs(img_paths, vis, model_dict, dataset_dict, draw_visualization_func=draw_all_vis_single_img)
