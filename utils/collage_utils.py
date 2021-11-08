import os

from PIL import Image, ImageDraw, ImageFont

from visualization.grad_cam_main import demo1

# Prepare fonts
FONT_SIZE_LABELS = 40
FONT_SIZE_VISUALIZATIONS = 30
FONT_SIZE_FILENAMES = 20
BOLD_FONT = 'fonts/RobotoCondensed-Bold.ttf'
BASIC_FONT = 'fonts/RobotoCondensed-Light.ttf'

# Set sizes
IMG_SIZE = 224
LABEL_X = IMG_SIZE # width of a box with a label
LABEL_Y = 80 # height of a box with a label
INTERFACING = 10 # spacing inbetween objects
PADDING = 20 # padding around the edges of a collage
VISUALIZATIONS = ['gradcam', 'guided_gradcam', 'vanilla', 'deconvolution']
CUDA = True

CLASSES = ['BLANK', 'NO CRYSTAL', 'WEAK', 'GOOD', 'STRONG']


def draw_text(text, dims, font=BASIC_FONT, font_size=FONT_SIZE_LABELS):
    """Draw text inside a box.

    Args:
        text (str): [description]
        position (tuple): , a tuple with 2 ints
        font ([type], optional): [description]. Defaults to BASIC_FONT.
        font_size ([type], optional): [description]. Defaults to FONT_SIZE_LABELS.

    Returns:
        [type]: [description]
    """
    textbox_img = Image.new('RGBA', dims, "#FFFFFF")
    textbox_draw = ImageDraw.Draw(textbox_img)
    textbox_draw.text((PADDING, INTERFACING), text, font= ImageFont.truetype(font, font_size), fill=(0,0,0,255))
    return textbox_img

    
def draw_caption(page, location, classes_list, model_name,  sample_info='', visualization=None):
    x, y = location

    # Draw title
    if type(sample_info) != str:
        title_text = draw_text(f'{model_name} {sample_info["id"]}', (LABEL_X*4, LABEL_Y))
    else:
        title_text = draw_text(f'{model_name} {visualization}', (LABEL_X*4, LABEL_Y))
    page.paste(title_text, (x, y))
    y += LABEL_Y

    # Draw label and class names
    if type(sample_info) != str:
        label_text = draw_text('label', (LABEL_X, LABEL_Y))
        page.paste(label_text, (0, y))
        x = INTERFACING + LABEL_X

    for c in classes_list:
        c_text = draw_text(c, (LABEL_X, LABEL_Y))
        page.paste(c_text, (x, y))
        x += INTERFACING + LABEL_X
    x = 0
    y += LABEL_Y

    # Draw label name and class prediction values,
    # highlight the value of the top prediction
    if type(sample_info) != str:
        label_val_text = draw_text(sample_info['label'], (LABEL_X*2, LABEL_Y), font=BOLD_FONT)
        page.paste(label_val_text, (x, y))
        x += INTERFACING + IMG_SIZE

        for idx, c in enumerate(classes_list):
            font = BOLD_FONT if sample_info['top_pred'] == idx else BASIC_FONT
            pred = f'{sample_info[str(idx)]:.3}'
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

    for vis in ['gradcam', 'guided_gradcam']:
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


def draw_vis_many_images(page, location, classes_list, results, img_paths, visualizations, draw_filename=False):
    x, y = location

    x += PADDING
    for class_idx, c in enumerate(classes_list):
        y = location[1]

        for img_path in img_paths[class_idx]:

            class_text = draw_text(classes_list[class_idx], (LABEL_X, LABEL_Y), font=BASIC_FONT, font_size=FONT_SIZE_LABELS)
            page.paste(class_text, (x, y))
            y += LABEL_Y

            result_imgs = results[img_path]

            for visualization in visualizations:
                if visualization == 'original':
                    image = result_imgs[visualization]
                elif visualization in ['guided_gradcam', 'vanilla', 'deconvolution']:
                    if result_imgs[visualization][class_idx].shape[2] == 1:
                    # if img is in grayscale
                        image = Image.fromarray(result_imgs[visualization][class_idx][:, :, 0])
                    else:
                        image = Image.fromarray(result_imgs[visualization][class_idx][:, :, 0], 'L')#, mode='L')# 'L')
                else:
                    image = Image.fromarray(result_imgs[visualization][class_idx])

                image = image.resize((224, 224))
                page.paste(image, (x, y))
                y += IMG_SIZE + INTERFACING

            if draw_filename:
                filename_text = draw_text(img_path, (LABEL_X, LABEL_Y), font_size=FONT_SIZE_FILENAMES)
                page.paste(filename_text, (x, y))
                y += LABEL_Y


        x += IMG_SIZE + INTERFACING
    
    y = location[1] + LABEL_Y + 50
    for visualization in visualizations:
        visualization_text = draw_text(visualization, (LABEL_X, LABEL_Y), font_size=FONT_SIZE_VISUALIZATIONS)
        page.paste(visualization_text, (x, y))
        y += IMG_SIZE + INTERFACING

    return page, x, y
