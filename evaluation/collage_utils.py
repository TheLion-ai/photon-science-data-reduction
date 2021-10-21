import os

from PIL import Image, ImageDraw, ImageFont

from utils.basic_utils import prepare_path
from visualization.grad_cam_main import demo1

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
VISUALIZATIONS = ['gradcam', 'guided_gradcam', 'vanilla', 'deconvolution']
CUDA = True


def draw_text(text, position, font=BASIC_FONT, font_size=FONT_SIZE_LABELS):
    textbox_img = Image.new('RGBA', position, "#FFFFFF")
    textbox_draw = ImageDraw.Draw(textbox_img)
    textbox_draw.text((PADDING, INTERFACING), text, font= ImageFont.truetype(font, font_size), fill=(0,0,0,255))
    return textbox_img


def draw_caption(page, location, classes_list, model_name,  sample_info=None, visualization=None):
    x, y = location

    # Draw title
    if sample_info:
        title_text = draw_text(f'{model_name} {sample_info["id"]}', (LABEL_X*4, LABEL_Y))
    else:
        title_text = draw_text(f'{model_name} {visualization}', (LABEL_X*4, LABEL_Y))
    page.paste(title_text, (x, y))
    y += LABEL_Y

    if sample_info:
        # Draw labels with predictions
        label_text = draw_text('label', (LABEL_X, LABEL_Y))
        page.paste(label_text, (0, y))
        x = INTERFACING + LABEL_X

    for c in classes_list:
        c_text = draw_text(c, (LABEL_X, LABEL_Y))
        page.paste(c_text, (x, y))
        x += INTERFACING + LABEL_X
    x = 0
    y += LABEL_Y

    if sample_info:
        label_val_text = draw_text(sample_info['label'], (LABEL_X*2, LABEL_Y), font=BOLD_FONT)
        page.paste(label_val_text, (x, y))
        x += INTERFACING + IMG_SIZE

        for idx, c in enumerate(classes_list):
            font = BOLD_FONT if sample_info['top_pred'] == idx else BASIC_FONT
            pred = f'{sample_info[c]:.3}'
            pred_text = draw_text(pred, (LABEL_X, LABEL_Y), font=font)
            page.paste(pred_text, (x, y))
            x += INTERFACING + IMG_SIZE
        y += LABEL_Y
        x = INTERFACING
    return page, x, y


def draw_all_vis_single_img(page, location, classes_list, sample_info, result_imgs,):

    x, y = location
    image = result_imgs['original']
    image = image.resize((IMG_SIZE, IMG_SIZE))
    page.paste(image, (x, y))

    for vis in VISUALIZATIONS:
        x = IMG_SIZE + INTERFACING*2

        for i in range(len(classes_list)):
            if vis in ['guided_gradcam', 'vanilla', 'deconvolution']:
                image = Image.fromarray(result_imgs[vis][i][:, :, 0], 'L')
            else:
                image = Image.fromarray(result_imgs[vis][i])
            image = image.resize((224, 224))
            page.paste(image, (x, y))
            x += IMG_SIZE + INTERFACING
        
        vis_text = draw_text(vis, (LABEL_X, LABEL_Y), font=BOLD_FONT, font_size=FONT_SIZE_VISUALIZATIONS)
        page.paste(vis_text, (x, y))
        y += IMG_SIZE + INTERFACING

    return page, x, y


def draw_vis_many_images(page, location, classes_list, results, img_paths, visualization):
    x, y = location

    x += PADDING
    for class_idx, c in enumerate(classes_list):
        y = location[1]
        # label_text = draw_text(c, (LABEL_X, LABEL_Y), font=BOLD_FONT, font_size=FONT_SIZE_LABELS)
        # page.paste(label_text, (x, y))
        # y += LABEL_Y

        for img_path in img_paths[class_idx]:
            result_imgs = results[img_path]
            if visualization in ['guided_gradcam', 'vanilla', 'deconvolution']:
                image = Image.fromarray(result_imgs[visualization][class_idx][:, :, 0], 'L')
            else:
                image = Image.fromarray(result_imgs[visualization][class_idx])

            image = image.resize((224, 224))
            page.paste(image, (x, y))
            y += IMG_SIZE + INTERFACING
        x += IMG_SIZE + INTERFACING

    # for c in classes_list:
    #     x = IMG_SIZE + INTERFACING*2

    #     for i in range(len(classes_list)):
            
    #         image = image.resize((224, 224))
    #         page.paste(image, (x, y))
    #         x += IMG_SIZE + INTERFACING
        
    #     vis_text = draw_text(visualization, (LABEL_X, LABEL_Y), font=BOLD_FONT, font_size=FONT_SIZE_VISUALIZATIONS)
    #     page.paste(vis_text, (x, y))
    #     y += IMG_SIZE + INTERFACING

    return page, x, y
