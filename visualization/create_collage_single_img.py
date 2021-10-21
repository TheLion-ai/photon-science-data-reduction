import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from utils.basic_utils import prepare_path
from visualization.grad_cam_main import demo1
from evaluation.collage_utils import *
from dicts.pretrained_models_dict import pretrained_models_dict
from dicts.datasets_dict import datasets_dict


def create_collage_single_img(img_paths, csv_set, model_dict, dataset_dict):
    
    backbone = model_dict['backbone']
    model_path = model_dict['path']
    target_layer = model_dict['target_layer']
    topk = model_dict['num_classes']
    num_classes = model_dict['num_classes']
    output_dir = prepare_path(model_dict['name'], dataset_dict['name'])
    classes_list = sorted(model_dict['classes'], key=model_dict['classes'].get)

    root_img_path = os.path.join('datasets', dataset_dict['name'], 'test')
    img_paths = [os.path.join(root_img_path, img_path) for img_path in img_paths]
    csv_path = os.path.join('prediction_csvs', model_dict['name'], dataset_dict['name'], f'{csv_set}.csv')

    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        class_idx = dataset_dict['classes'][row['label']]
        img_path = os.path.join(root_img_path, str(class_idx), row['id'])
        results = demo1([img_path], target_layer, topk, output_dir, CUDA, backbone, model_path, num_classes)
        
        x, y = (0, 0)
        page = Image.new('RGB', (LABEL_X+6*IMG_SIZE+INTERFACING*5+PADDING*2, 4*IMG_SIZE+3*INTERFACING+LABEL_Y*3+PADDING), '#FFFFFF')
        page, x, y = draw_caption(page, (x, y), classes_list, row, model_dict["name"])

        result_imgs = results[row['id']]
        draw_all_vis_single_img(page, (x, y), classes_list, row, result_imgs)
            
        page.save(os.path.join(output_dir, f'{model_dict["name"]}_{row["id"][:-4]}_{csv_set}.png'))


if __name__ == '__main__':
    csv_set = 'undecided' #'undecided' #'misclassified'
    img_paths = {
        '0': ['fake_19166.png', 'fake_12408.png', 'fake_12413.png', 'fake_12428.png', 'fake_12429.png'],
        '1': ['fake_19635.png', 'fake_12401.png', 'fake_12402.png', 'fake_12404.png', 'fake_12410.png'],
        '2': ['fake_17109.png', 'fake_12403.png', 'fake_12406.png', 'fake_12412.png', 'fake_12417.png'],
        '3': ['fake_18850.png', 'fake_12405.png', 'fake_12409.png', 'fake_12411.png', 'fake_12415.png'],
        '4': ['fake_13532.png', 'fake_12407.png', 'fake_12418.png', 'fake_12419.png', 'fake_12424.png']
    }

    dataset_dict = datasets_dict['diffranet']['synthetic']

    model_dict = pretrained_models_dict['vgg19_all_layers']
    create_collage_single_img(img_paths, csv_set, model_dict, dataset_dict)
