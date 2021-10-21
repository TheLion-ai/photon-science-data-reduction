import os

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from dicts.pretrained_models_dict import pretrained_models_dict
from visualizations.grad_cam_main import demo1

bold_font = '/home/bk/sfx/model/fonts/RobotoCondensed-Bold.ttf'
basic_font = '/home/bk/sfx/model/fonts/RobotoCondensed-Light.ttf'

# classes = {
#     0: {
#         'name': 'blank',
#         'position': (200, 40)
#         },
#     1: {
#         'name': 'no-crystal',
#         'position': (320, 40)
#         },
#     2: {
#         'name': 'weak',
#         'position': (450, 40)
#         },
#     3: {
#         'name': 'good',
#         'position': (570, 40)
#         },
#     4: {
#         'name': 'strong',
#         'position': (690, 40)
#         }
# }
# Prepare fonts
FONT_SIZE_LABELS = 40
FONT_SIZE_VISUALIZATIONS = 30
BOLD_FONT = 'fonts/RobotoCondensed-Bold.ttf'
BASIC_FONT = 'fonts/RobotoCondensed-Light.ttf'

# Set sizes
IMG_SIZE = 224
LABEL_X = IMG_SIZE
LABEL_Y = 80
INTERFACING = 10
PADDING = 20


def draw_text(text, position, font=BASIC_FONT, font_size=FONT_SIZE_LABELS):
    textbox_img = Image.new('RGBA', position, "#FFFFFF")
    textbox_draw = ImageDraw.Draw(textbox_img)
    textbox_draw.text((PADDING, INTERFACING), text, font= ImageFont.truetype(font, font_size), fill=(0,0,0,255))
    return textbox_img

root_img_path = os.path.join('datasets/diffranet/synthetic/test')
img_paths = [
    '0/fake_19166.png',
    '1/fake_19635.png',
    '2/fake_17109.png',
    '3/fake_18850.png',
    '4/fake_13532.png'
]
img_paths = [os.path.join(root_img_path, img_path) for img_path in img_paths]

# for model in pretrained_models_dict.values():
model = pretrained_models_dict['vgg19_full_tune_imagenet']
target_layer = model['target_layer']
num_classes = model['num_classes']
topk = num_classes
backbone = model['backbone']
model_path = model['path']
output_dir = 'results/'
cuda = True

result_imgs = demo1(img_paths, target_layer, topk, output_dir, cuda, backbone, model_path, num_classes)

visualizations = ['gradcam', 'guided_gradcam', 'vanilla', 'deconvolution']


for vis in visualizations:
    page = Image.new('RGB', (1668, 1400), '#FFFFFF')

    textbox_img = Image.new('RGBA', (224*2, 80), "#FFFFFF")
    textbox_draw = ImageDraw.Draw(textbox_img)
    textbox_draw.text((50, 10), f'{model["backbone"].__name__} {vis}', font= ImageFont.truetype(basic_font, 40), fill=(0,0,0,255))
    page.paste(textbox_img, (0, 0))

    textbox_img = Image.new('RGBA', (224, 80), "#FFFFFF")
    textbox_draw = ImageDraw.Draw(textbox_img)
    textbox_draw.text((50, 10), 'label', font= ImageFont.truetype(basic_font, 40), fill=(0,0,0,255))
    page.paste(textbox_img, (0, 80))

    textbox_img = Image.new('RGBA', (1668-224*2, 80), "#FFFFFF")
    textbox_draw = ImageDraw.Draw(textbox_img)
    textbox_draw.text((224, 10), 'predicted class', font= ImageFont.truetype(basic_font, 40), fill=(0,0,0,255))
    page.paste(textbox_img, (224*2, 0))

    textbox_img = Image.new('RGBA', (234, 80), "#FFFFFF")
    textbox_draw = ImageDraw.Draw(textbox_img)
    textbox_draw.text((10, 10), 'input', font= ImageFont.truetype(basic_font, 40), fill=(0,0,0,255))
    page.paste(textbox_img, (300, 80))
    x = 234*2

    for c in classes:
        textbox_img = Image.new('RGBA', (234, 80), "#FFFFFF")
        textbox_draw = ImageDraw.Draw(textbox_img)
        textbox_draw.text((40, 10), classes[c]['name'], font= ImageFont.truetype(basic_font, 40), fill=(0,0,0,255))
        page.paste(textbox_img, (x, 80))
        x += 234


    
    y = 2*80
    for idx, img in enumerate(result_imgs.values()):
        x = 0

        textbox_img = Image.new('RGBA', (224, 224), "#FFFFFF")
        textbox_draw = ImageDraw.Draw(textbox_img)
        textbox_draw.text((50, 100), classes[idx]['name'], font= ImageFont.truetype(basic_font, 40), fill=(0,0,0,255))
        page.paste(textbox_img, (x, y))
        x += 234

        image = img['original']
        imag = image.resize((224, 224))
        page.paste(imag, (x, y))
        x += 234
        for i in range(num_classes):
            if vis in ['guided_gradcam', 'vanilla', 'deconvolution']:
                image = Image.fromarray(img[vis][i][:, :, 0], 'L')
            else:
                image = Image.fromarray(img[vis][i])
            imag = image.resize((224, 224))
            page.paste(imag, (x, y))
            x += 234
        y += 234
    page.save(f'{model["backbone"].__name__}_fc_{vis}.png')


