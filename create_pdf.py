from operator import inv
import os

import pandas as pd
from PIL import Image, ImageFont, ImageDraw

from torch import load
from torch.nn import Linear  
from torch.nn.functional import softmax  
from torchvision.models.alexnet import AlexNet

from pytorch_cnn_visualizations.src.gradcam import GradCam
from pytorch_cnn_visualizations.src.guided_backprop import GuidedBackprop
from pytorch_cnn_visualizations.src.inverted_representation import InvertedRepresentation
from pytorch_cnn_visualizations.src.guided_gradcam import guided_grad_cam
from pytorch_cnn_visualizations.src.misc_functions import apply_colormap_on_image, preprocess_image, format_np_output, convert_to_grayscale

from datasets_dict import datasets_dict

num_classes = 2
model_path = '/home/bk/sfx/logs/alexnet_fine_tune_diffranet-epoch=22-val_loss=0.22.ckpt'

root_dir_new_imgs = 'model/visualization_results'

bold_font = '/home/bk/sfx/model/fonts/RobotoCondensed-Bold.ttf'
basic_font = '/home/bk/sfx/model/fonts/RobotoCondensed-Light.ttf'

# classes = {
#     'blank': {
#         'class_no': '0',
#         'position': (200, 40)
#         },
#     'no-crystal': {
#         'class_no': '1',
#         'position': (320, 40)
#         },
#     'weak': {
#         'class_no': '2',
#         'position': (450, 40)
#         },
#     'good': {
#         'class_no': '3',
#         'position': (570, 40)
#         },
#     'strong': {
#         'class_no': '4',
#         'position': (690, 40)
#         }
# }

classes = {
    'no_diffraction': {
        'class_no': '0',
        'position': (200, 40)
        },
    'diffraction': {
        'class_no': '1',
        'position': (450, 40)
        }
}


def get_blank_page(word):
    page = Image.new('RGB', (1280, 720), '#37474F')
    textbox_img = Image.new('RGBA', (1180, 100), '#37474F')
    # put text on image
    textbox_draw = ImageDraw.Draw(textbox_img)
    textbox_draw.text((20, 10), word, font= ImageFont.truetype(bold_font, 60), fill=(255,255,255,255))
    page.paste(textbox_img, (200, 300))
    return page


def generate_visualizations_pdf(csv_paths):
    pages = []
    

    model = load_trained_alexnet(model_path, num_classes)
    # get gradcam image
    grad_cam = GradCam(model, target_layer=11)
    GBP = GuidedBackprop(model)
    inverted_representation = InvertedRepresentation(model)

    

    page = get_blank_page("AlexNet fine-tuned on Diffranet")
    pages.append(page)
    words = ['CERTAIN', 'UNDECIDED', 'MISCLASSIFIED']
    datasets = datasets_dict.keys()

    for dataset in datasets:
        pages = []
        page = get_blank_page(dataset)
        pages.append(page)
        root_dir = datasets_dict[dataset]['path'] + '/test'
        for csv, word in zip(csv_paths[dataset], words):
            df = pd.read_csv(csv)
            page = get_blank_page(word)
            pages.append(page)

            for index, row in df.iterrows():
                # if row['label'] == 'no_diffraction':
                #     c = '0'
                # elif row['label']=='diffraction':
                #     c = '1'
                # else:
                c = classes[row['label']]['class_no']

                img_path = os.path.join(root_dir, c, row['id'])
                page = Image.new('RGB', (1280, 720), '#37474F')
                img_base = Image.open(img_path)
                page.paste(img_base, (10, 150))

                prep_img_base = preprocess_image(img_base.convert('RGB'))

                # Generate cam mask
                cam = grad_cam.generate_cam(prep_img_base, row['top_pred'])
                heatmap, heatmap_on_image = apply_colormap_on_image(img_base, cam, 'hsv')
                print(index)

                # print('Grad cam completed')
                cam_resized = format_np_output(cam)
                cam_resized = Image.fromarray(cam_resized)
                cam_resized = cam_resized.resize((300, 300))
                page.paste(cam_resized, (970, 100))
                heatmap_on_image = heatmap_on_image.resize((300, 300))
                page.paste(heatmap_on_image, (970, 420))


                # get guided backprop
                guided_grads = GBP.generate_gradients(prep_img_base, row['top_pred'])
                cam_gb = guided_grad_cam(cam, guided_grads)
                gray_cam_gb = convert_to_grayscale(cam_gb)

                gray_cam_gb = gray_cam_gb - gray_cam_gb.min()
                gray_cam_gb /= gray_cam_gb.max()
                gray_cam_gb = format_np_output(gray_cam_gb)
                gray_cam_gb = Image.fromarray(gray_cam_gb)
                gray_cam_gb = gray_cam_gb.resize((300, 300))
                page.paste(gray_cam_gb, (600, 100)) #210
                
                cam_gb = cam_gb - cam_gb.min()
                cam_gb /= cam_gb.max()
                cam_gb = format_np_output(cam_gb)
                cam_gb = Image.fromarray(cam_gb)
                cam_gb = cam_gb.resize((300, 300))
                page.paste(cam_gb, (600, 420)) #210
                
                textbox_img = Image.new('RGBA', (1260, 80), "#E0E0E0")
                # put text on image
                textbox_draw = ImageDraw.Draw(textbox_img)
                # textbox_draw.text((20, 20), "LABEL                             BLANK         NO-CRYSTAL           WEAK                 GOOD                STRONG", font= ImageFont.truetype(basic_font, 20), fill=(0,0,0,255))
                textbox_draw.text((20, 20), "LABEL                             NO DIFFRACTION                     DIFFRACTION", font= ImageFont.truetype(basic_font, 20), fill=(0,0,0,255))
                textbox_draw.text((20, 40), row['label'], font= ImageFont.truetype(bold_font, 20), fill=(0,0,0,255))
                
                for c in classes.keys():
                    key = classes[c]
                    font = bold_font if str(row['top_pred']) == key['class_no'] else basic_font
                    num = row[c]
                    row[c] = f'{num:.3}'
                    textbox_draw.text(key['position'], row[c], font= ImageFont.truetype(font, 20), fill=(0,0,0,255))

                page.paste(textbox_img, (10, 10))

                new_img_path = os.path.join(root_dir_new_imgs, f"{row['id']}")

                # page.save(new_img_path)
                pages.append(page)

                # page 2 - inverting visual representations
                page = Image.new('RGB', (1280, 720), '#37474F')
                textbox_img = Image.new('RGBA', (1260, 80), "#E0E0E0")
                # put text on image
                textbox_draw = ImageDraw.Draw(textbox_img)
                # textbox_draw.text((20, 20), "LABEL                             BLANK         NO-CRYSTAL           WEAK                 GOOD                STRONG", font= ImageFont.truetype(basic_font, 20), fill=(0,0,0,255))
                textbox_draw.text((20, 20), "LABEL                             NO DIFFRACTION                     DIFFRACTION", font= ImageFont.truetype(basic_font, 20), fill=(0,0,0,255))
                textbox_draw.text((20, 40), row['label'], font= ImageFont.truetype(bold_font, 20), fill=(0,0,0,255))
                for c in classes.keys():
                    key = classes[c]
                    font = bold_font if str(row['top_pred']) == key['class_no'] else basic_font
                    num = row[c]
                    row[c] = f'{num:.3}'
                    textbox_draw.text(key['position'], row[c], font= ImageFont.truetype(font, 20), fill=(0,0,0,255))
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



        p = pages[0]
        pages = pages[1:]
        p.save(f'Alex Net_fine_tune_diffranet_{dataset}1.pdf',save_all=True, append_images=pages)


if __name__ == '__main__':
    csv_paths = {
        # 'diffranet_synthetic': [
        #     '/home/bk/sfx/model/prediction_csvs/alexnet_fine_tune_diffranet/diffranet_synthetic/certain.csv',
        #     '/home/bk/sfx/model/prediction_csvs/alexnet_fine_tune_diffranet/diffranet_synthetic/undecided.csv',
        #     '/home/bk/sfx/model/prediction_csvs/alexnet_fine_tune_diffranet/diffranet_synthetic/misclassified.csv'
        #     ],
        'diffranet_real_preprocessed': [
            '/home/bk/sfx/model/prediction_csvs/alexnet_fine_tune_diffranet/diffranet_real_preprocessed/certain.csv',
            '/home/bk/sfx/model/prediction_csvs/alexnet_fine_tune_diffranet/diffranet_real_preprocessed/undecided.csv',
            '/home/bk/sfx/model/prediction_csvs/alexnet_fine_tune_diffranet/diffranet_real_preprocessed/misclassified.csv'
            ],
        'diffranet_real_raw':  [
            '/home/bk/sfx/model/prediction_csvs/alexnet_fine_tune_diffranet/diffranet_real_raw/certain.csv',
            '/home/bk/sfx/model/prediction_csvs/alexnet_fine_tune_diffranet/diffranet_real_raw/undecided.csv',
            '/home/bk/sfx/model/prediction_csvs/alexnet_fine_tune_diffranet/diffranet_real_raw/misclassified.csv'
            ],
        
    }
    generate_visualizations_pdf(csv_paths)