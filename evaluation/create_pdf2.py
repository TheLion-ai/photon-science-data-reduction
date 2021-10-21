from operator import inv
import os

import pandas as pd
from PIL import Image, ImageFont, ImageDraw

from torch import load
from torch.nn import Linear  
from torch.nn.functional import softmax  
from torchvision.models.alexnet import AlexNet

from utils.model_utils import load_trained_model
from lenet import LeNet
from pytorch_cnn_visualizations.src.gradcam import GradCam
from pytorch_cnn_visualizations.src.guided_backprop import GuidedBackprop
from pytorch_cnn_visualizations.src.inverted_representation import InvertedRepresentation
from pytorch_cnn_visualizations.src.guided_gradcam import guided_grad_cam
from pytorch_cnn_visualizations.src.misc_functions import apply_colormap_on_image, preprocess_image, format_np_output, convert_to_grayscale

from dicts.datasets_dict import datasets_dict
from dicts.pretrained_models_dict import pretrained_models_dict
from utils.data_utils import get_transforms

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

# classes = {
#     0: {
#         'name': 'no_diffraction',
#         'position': (200, 40)
#         },
#     1: {
#         'name': 'diffraction',
#         'position': (450, 40)
#         }
# }


def get_blank_page(word):
    # Create a blank page with a title
    page = Image.new('RGB', (1280, 720), '#37474F')
    textbox_img = Image.new('RGBA', (1180, 100), '#37474F')
    # put text on image
    textbox_draw = ImageDraw.Draw(textbox_img)
    textbox_draw.text((20, 10), word, font= ImageFont.truetype(bold_font, 60), fill=(255,255,255,255))
    page.paste(textbox_img, (200, 300))
    return page




def generate_visualizations_pdf(model_name, backbone, model_path, num_classes, 
                                dataset, verbose=True):
    # TODO: Add img_size
    pages = []
    model = load_trained_model(backbone, model_path, num_classes, img_size)

    inverted_representation = InvertedRepresentation(model)

    page = get_blank_page(f'{backbone.__name__}')
    pages.append(page)
    words = ['CERTAIN', 'UNDECIDED', 'MISCLASSIFIED']

    pages = []
    page = get_blank_page(dataset['name'])
    pages.append(page)
    root_dir = dataset['path'] + '/test'
    img_paths = [
    '0/fake_19166.png',
    '1/fake_19635.png',
    '2/fake_17109.png',
    '3/fake_18850.png',
    '4/fake_13532.png'
    ]
    root_img_path = os.path.join('datasets/diffranet/synthetic/test')


    for img_path in img_paths:


        img_base = Image.open(os.path.join(root_img_path, img_path))
        page = Image.new('RGB', (1280, 720), '#37474F')

        prep_img_base = preprocess_image(img_base.convert('RGB'))
        # my_transforms = get_transforms(224, True)
        # prep_img_base = my_transforms(img_base)


        page = Image.new('RGB', (1280, 720), '#37474F')


        img_base = img_base.resize((200, 200))
        page.paste(img_base, (10, 100))
        new_img_path = os.path.join(root_dir_new_imgs, f"{img_path[4:]}")
        for i in range(36): # features
            inv_img = inverted_representation.generate_inverted_image_specific_layer(prep_img_base, 224, True, i, img_path[4:])
            inv_img = Image.fromarray(inv_img)
            inv_img = inv_img.resize((100, 100))
            j = i % 10
            w = 10 + 110*(j+1)
            h = 100 + 110 * (i // 5)
            page.paste(inv_img, (w, h)) #210

        new_img_path = os.path.join(root_dir_new_imgs, f"{img_path[4:]}")

        page.save(new_img_path)


if __name__ == '__main__':

    model_name = 'vgg19_full_tune_imagenet'
    pretrained_model = pretrained_models_dict[model_name]
    backbone = pretrained_model['backbone']
    model_path = pretrained_model['path']
    num_classes = pretrained_model['num_classes']
    dataset = datasets_dict['diffranet']['synthetic']
    generate_visualizations_pdf(model_name, backbone, model_path, num_classes, dataset) 

    root_dir_new_imgs = 'results1'
    model_name = 'vgg19_fine_tune_imagenet'
    pretrained_model = pretrained_models_dict[model_name]
    backbone = pretrained_model['backbone']
    model_path = pretrained_model['path']
    num_classes = pretrained_model['num_classes']
    dataset = datasets_dict['diffranet']['synthetic']
    generate_visualizations_pdf(model_name, backbone, model_path, num_classes, dataset) 