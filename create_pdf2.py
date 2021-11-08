from operator import inv
import os

import pandas as pd
from PIL import Image, ImageFont, ImageDraw

from torch import load
from torch.nn import Linear  
from torch.nn.functional import softmax  
from torchvision.models.alexnet import AlexNet

from utils.model_utils import load_trained_model
from model.lenet import LeNet
from visualization.inverted_representation import InvertedRepresentation
from visualization.misc_functions import *
from utils.collage_utils import *

from dicts.datasets_dict import datasets_dict
from dicts.pretrained_models_dict import pretrained_models_dict

num_classes = 5

root_dir_new_imgs = 'results'

bold_font = '/home/bk/sfx/model/fonts/RobotoCondensed-Bold.ttf'
basic_font = '/home/bk/sfx/model/fonts/RobotoCondensed-Light.ttf'

classes = {
    0: {
        'name': 'blank',
        'position': (200, 40)
        },
    1: {
        'name': 'no-crystal',
        'position': (320, 40)
        },
    2: {
        'name': 'weak',
        'position': (450, 40)
        },
    3: {
        'name': 'good',
        'position': (570, 40)
        },
    4: {
        'name': 'strong',
        'position': (690, 40)
        }
}



def get_blank_page(word):
    # Create a blank page with a title
    page = Image.new('RGB', (1280, 720), '#37474F')
    textbox_img = Image.new('RGBA', (1180, 100), '#37474F')
    # put text on image
    textbox_draw = ImageDraw.Draw(textbox_img)
    textbox_draw.text((20, 10), word, font= ImageFont.truetype(bold_font, 60), fill=(255,255,255,255))
    page.paste(textbox_img, (200, 300))
    return page




def generate_visualizations_pdf(model_dict, dataset_dict, verbose=True):
    # model_name, backbone, model_path, num_classes, 
    #                             dataset, verbose=True):
    # TODO: Add img_size
    pages = []
    model = load_trained_model(model_dict['backbone'], model_dict['path'], model_dict['num_classes'], model_dict['img_size'])

    inverted_representation = InvertedRepresentation(model)

    page = get_blank_page(f'{model_dict["backbone"].__name__}')
    pages.append(page)
    words = ['CERTAIN', 'UNDECIDED', 'MISCLASSIFIED']

    pages = []
    page = get_blank_page(dataset_dict['name'])
    pages.append(page)
    root_dir = os.path.join(dataset_dict['path'], dataset_dict['test_dir'])
    img_paths = [
    # '0/fake_19166.png',
    # '1/fake_19635.png',
    '2/fake_17109.png',
    # '3/fake_18850.png',
    # '4/fake_13532.png'
    ]
    root_img_path = os.path.join('datasets/diffranet_synthetic/test')


    for img_path in img_paths:


        img_base = Image.open(os.path.join(root_img_path, img_path))
        # page = Image.new('RGB', (12--, 820))# '#37474F')

        prep_img_base = preprocess_image(img_base.convert('RGB'))
        # my_transforms = get_transforms(224, True)
        # prep_img_base = my_transforms(img_base)


        page = Image.new('RGB', (210*5+10, 250*3), '#FFFFFF')

        input_text = draw_text('Input', (200, 40), font_size=FONT_SIZE_FILENAMES)
        page.paste(input_text, (10, 0))
        img_base = img_base.resize((200, 200))
        page.paste(img_base, (10, 40))
        new_img_path = os.path.join(root_dir_new_imgs, f"{img_path[4:]}")
        for i in range(len(model_dict['layers'])-1): # features
            inv_img = inverted_representation.generate_inverted_image_specific_layer(prep_img_base, 224, i) #, img_path[4:])
            inv_img = Image.fromarray(inv_img)
            inv_img = inv_img.resize((200, 200))
            j = i % 4
            w = 10 + 210*(j+1)
            h = 0 + 250*(i// 4) #+ 250 #* (i// 6)

            filename_text = draw_text(model_dict['layers'][i], (200, 40), font_size=FONT_SIZE_FILENAMES)
            page.paste(filename_text, (w, h))

            page.paste(inv_img, (w, h+40)) #210


        new_img_path = os.path.join(root_dir_new_imgs, f"{img_path[4:]}")

        page.save(new_img_path)



if __name__ == '__main__':

    model_dict = pretrained_models_dict['alexnet_all_layers']
    dataset_dict = datasets_dict['diffranet']['synthetic']

    generate_visualizations_pdf(model_dict, dataset_dict)