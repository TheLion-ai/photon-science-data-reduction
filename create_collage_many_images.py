import os

import numpy as np
from numpy.core.numeric import full
import pandas as pd

from PIL import Image, ImageDraw, ImageFont

from utils.basic_utils import prepare_results_path
from visualization.grad_cam_main import demo1
from utils.collage_utils import *
from dicts.pretrained_models_dict import pretrained_models_dict
from dicts.datasets_dict import datasets_dict


def create_visulizations_collage_many_imgs(img_paths, vis, model_dict, dataset_dict, draw_visualization_func, img_idx=0, caption=False):
    
    output_dir = prepare_results_path(model_dict['name'], dataset_dict['name'])
    classes_list = sorted(model_dict['classes'], key=model_dict['classes'].get)

    root_img_path = os.path.join('datasets', dataset_dict['name'], dataset_dict['test_dir'])
    

    # TODO global img)paths in demo1
    results = {}
    full_img_paths = []
    for key in img_paths:
        full_img_paths.extend([os.path.join(root_img_path, str(key), img_path) for img_path in img_paths[key]])

    results = demo1(full_img_paths, model_dict, visualizations=[vis])
    # results = {**results, **result}

    
    x, y = (0, 10)
    page = Image.new('RGB', (6*IMG_SIZE+INTERFACING*4+PADDING*2, (len(vis)-1)*IMG_SIZE+2*INTERFACING+LABEL_Y*(2+2)), '#FFFFFF')

    # page, x, y = draw_caption(page, (x, y), classes_list, model_dict['name'], visualization=vis)
    page, x, y = draw_vis_many_images(page, (x, y), classes_list, results, img_paths, vis)
            
    page.save(os.path.join(output_dir, f'{model_dict["name"]}_{vis}_collage_many_imgs_{img_idx}.png'))


if __name__ == '__main__':

    dataset_dict = datasets_dict['diffranet']['synthetic']

    img_paths = {
        0: ['fake_12945.png'],
        1: ['fake_15622.png'],
        2: ['fake_16019.png'], #16019
        3: ['fake_24802.png'], #15443 23217
        4: ['fake_24725.png'] #21399
    }

    if not img_paths:
        all_img_paths = {}
        for key in img_paths.keys():
            path = os.path.join(dataset_dict['path'], 'test', str(key))
            filepaths = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            all_img_paths[key] = filepaths

    print('done')

    model_dict = pretrained_models_dict['alexnet_fc']
    vis = ['original', 'gradcam', 'guided_gradcam']
    # for i in range(20):
    # img_paths = {}
    # i = 2
    # for key in all_img_paths.keys():
    #     img_paths[key] = all_img_paths[key][i*3:(i+1)*3]
    create_visulizations_collage_many_imgs(img_paths, vis, model_dict, dataset_dict, draw_visualization_func=draw_all_vis_single_img)
