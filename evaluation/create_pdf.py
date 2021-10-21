from data_utils import get_transforms
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
from pytorch_cnn_visualizations.src.gradcam import GradCam
from pytorch_cnn_visualizations.src.guided_backprop import GuidedBackprop
from pytorch_cnn_visualizations.src.inverted_representation import InvertedRepresentation
from pytorch_cnn_visualizations.src.guided_gradcam import guided_grad_cam
from pytorch_cnn_visualizations.src.misc_functions import apply_colormap_on_image, preprocess_image, format_np_output, convert_to_grayscale

from dicts.datasets_dict import datasets_dict
from dicts.pretrained_models_dict import pretrained_models_dict

num_classes = 5
backbone = LeNet
model_path = "/home/bk/sfx/model/logs/32_2009_2052LeNet_tune_fc_onlyFalse_{'name': 'diffranet_synthetic', 'path': 'datasets/diffranet/synthetic', 'multilabel': False, 'train_dir': 'train', 'val_dir': 'val', 'num_classes': 5, 'classes': {0: 'blank', 1: 'no-crystal', 2: 'weak', 3: 'good', 4: 'strong'}}/-epoch=49-val_loss=0.37-accuracy=0.00.ckpt"

root_dir_new_imgs = 'model/visualization_results'

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


def create_caption(label, top_pred, class_preds, num_classes):
    # Add label and class probs
    textbox_img = Image.new('RGBA', (1260, 80), "#E0E0E0")
    # put text on image
    textbox_draw = ImageDraw.Draw(textbox_img)
    if num_classes == 5:
        textbox_draw.text((20, 20), "LABEL                             BLANK         NO-CRYSTAL           WEAK                 GOOD                STRONG", font= ImageFont.truetype(basic_font, 20), fill=(0,0,0,255))
    elif num_classes == 2:
        textbox_draw.text((20, 20), "LABEL                             NO DIFFRACTION                     DIFFRACTION", font= ImageFont.truetype(basic_font, 20), fill=(0,0,0,255))
    
    textbox_draw.text((20, 40), label, font= ImageFont.truetype(bold_font, 20), fill=(0,0,0,255))
    
    # Write probbilities for each classes
    for idx, pred in enumerate(class_preds):
        font = bold_font if top_pred == idx else basic_font
        pred = f'{pred:.3}'
        position = classes[idx]['position']
        textbox_draw.text(position, pred, font= ImageFont.truetype(font, 20), fill=(0,0,0,255))
        return textbox_img

def get_gradcam(grad_cam, img_base, prep_img_base, top_pred):
    cam = grad_cam.generate_cam(prep_img_base, top_pred)
    heatmap, heatmap_on_image = apply_colormap_on_image(img_base, cam, 'hsv')
    cam_resized = format_np_output(cam)
    cam_resized = Image.fromarray(cam_resized)
    cam_resized = cam_resized.resize((300, 300))
    heatmap_on_image = heatmap_on_image.resize((300, 300))

    return cam, cam_resized, heatmap_on_image


def get_guided_grad_cam(GBP, prep_img_base, cam, top_pred):
    guided_grads = GBP.generate_gradients(prep_img_base, top_pred)
    cam_gb = guided_grad_cam(cam, guided_grads)
    gray_cam_gb = convert_to_grayscale(cam_gb)

    gray_cam_gb = gray_cam_gb - gray_cam_gb.min()
    gray_cam_gb /= gray_cam_gb.max()
    gray_cam_gb = format_np_output(gray_cam_gb)
    gray_cam_gb = Image.fromarray(gray_cam_gb)
    gray_cam_gb = gray_cam_gb.resize((300, 300))
    
    cam_gb = cam_gb - cam_gb.min()
    cam_gb /= cam_gb.max()
    cam_gb = format_np_output(cam_gb)
    cam_gb = Image.fromarray(cam_gb)
    cam_gb = cam_gb.resize((300, 300))
    
    return gray_cam_gb, cam_gb


def generate_visualizations_pdf(model_name, backbone, model_path, num_classes, 
                                dataset, verbose=True):

    # TODO ADD IMG_SIZE
    pages = []
    model = load_trained_model(backbone, model_path, num_classes, img_size)

    # load GradCAM, GuidedGradCAM and Inverted Representation models
    grad_cam = GradCam(model, target_layer=11)
    GBP = GuidedBackprop(model)
    inverted_representation = InvertedRepresentation(model)

    page = get_blank_page(f'{backbone.__name__}')
    pages.append(page)
    words = ['CERTAIN', 'UNDECIDED', 'MISCLASSIFIED']

    pages = []
    page = get_blank_page(dataset['name'])
    pages.append(page)
    root_dir = dataset['path'] + '/test'

    for word in words:
        csv_path = os.path.join('prediction_csvs', model_name, dataset['name'], f'{word.lower()}.csv')
        df = pd.read_csv(csv_path)
        page = get_blank_page(word)
        pages.append(page)

        for index, row in df.iterrows():

            if dataset['num_classes'] != num_classes:
                if num_classes == 2:
                    if row['label'] in ['blank', 'no_crystal']:
                        c = 0
                    else:
                        c = 1

            else:
                c = list({ k for k,v in classes.items() if v['name'] == row['label']})[0]


            # load base img and put it in a page
            img_path = os.path.join(root_dir, str(c), row['id'])
            img_base = Image.open(img_path)
            page = Image.new('RGB', (1280, 720), '#37474F')
            page.paste(img_base, (10, 150))

            # Prepare img for visualizations
            prep_img_base = preprocess_image(img_base.convert('RGB'))

            # Generate GradCAM image
            cam, cam_resized, heatmap_on_image = get_gradcam(grad_cam, prep_img_base, row['top_pred'])
            page.paste(cam_resized, (970, 100))
            page.paste(heatmap_on_image, (970, 420))
            if verbose:
                print('Grad cam completed')


            # Create Guided backpropagation img
            gray_cam_gb, cam_gb = get_guided_grad_cam(GBP, prep_img_base, cam, row['top_pred'])
            page.paste(gray_cam_gb, (600, 100)) #210
            page.paste(cam_gb, (600, 420)) #210
            
            
            class_preds = [row[classes[i]['name']] for i in range(num_classes)]
            textbox_img = create_caption(row['label'], row['top_pred'], class_preds, num_classes)

            page.paste(textbox_img, (10, 10))

            # # Code for saving a single img
            # new_img_path = os.path.join(root_dir_new_imgs, f"{row['id']}")
            # page.save(new_img_path)
            pages.append(page)

            # ----------------------------------------
            # page 2 - inverting visual representations
            page = Image.new('RGB', (1280, 720), '#37474F')


            page.paste(textbox_img, (10, 10))
            img_base = img_base.resize((200, 200))
            page.paste(img_base, (10, 100))
            for i in range(12):
                inv_img = inverted_representation.generate_inverted_image_specific_layer(prep_img_base, 224, i, row['id'])
                inv_img = Image.fromarray(inv_img)
                inv_img = inv_img.resize((200, 200))
                j = i % 5
                w = 10 + 210*(j+1)
                h = 100 + 210 * (i // 5)
                page.paste(inv_img, (w, h)) #210
                new_img_path = os.path.join(root_dir_new_imgs, f"inv_{row['id']}")
            # page.save(new_img_path)
            pages.append(page)
            break



        p = pages[0]
        # pages = pages[1:]
        p.save(f'alexnet_1p_{dataset["name"]}.pdf', save_all=True, append_images=pages)


if __name__ == '__main__':

    model_name = 'alexnet_full_tune_imagenet'
    pretrained_model = pretrained_models_dict[model_name]
    backbone = pretrained_model['backbone']
    model_path = pretrained_model['path']
    num_classes = pretrained_model['num_classes']
    dataset = datasets_dict['diffranet']['synthetic']
    generate_visualizations_pdf(model_name, backbone, model_path, num_classes, dataset)