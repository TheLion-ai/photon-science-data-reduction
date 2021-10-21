import os

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torch import load
from torch.nn import Linear  
from torch.nn.functional import softmax  
from torchvision.models import alexnet, vgg19

from dicts.datasets_dict import datasets_dict
from utils.data_utils import get_transforms
from utils.model_utils import load_trained_model, get_single_prediction
from model.lenet import LeNet

# def get_single_prediction(img_path, model):
#     img = Image.open(img_path).convert('RGB')
#     my_transforms = get_transforms(img_size=224, grayscale=False) # TODO: Remove hardcoding
#     x = my_transforms(img)
#     preds = model(x.unsqueeze(0))
#     preds = softmax(preds[0], dim=0)
#     np_preds = preds.detach().numpy()
#     top_pred = list(np_preds).index(np_preds.max())

#     return preds, np_preds, top_pred


def get_prediction_csvs(model_dict, dataset_dict, csv_path):
# Creates a separate csv for images from categories: correct, certain, misclassified and undecided
# and a joint csv for all of the above
    model_path = model_dict['path']
    backbone = model_dict['backbone']
    num_classes = model_dict['num_classes']
    img_size = model_dict['img_size']
    grayscale = model_dict['grayscale']

    model = load_trained_model(backbone, model_path, num_classes, img_size, fine_tune=True)
    root_path = os.path.join(dataset_dict['path'], dataset_dict['test_dir'])

    columns = ['id', 'label', 'top_pred'] + list(datasets_dict['diffranet']['synthetic']['classes'].values())
    columns_all = columns + ['correct', 'certain', 'misclassified', 'undecided']
    rows_all = []
    rows_certain = []
    rows_misclassified = []
    rows_undecided = []
    rows_correct=[]

    for dirr in os.listdir(root_path):
        dir_path = os.path.join(root_path, dirr)
        filenames = os.listdir(dir_path)
        
        # # Limit the results to 10 images
        # if len(filenames)> 11:
        #     filenames = filenames[0:10]
        for filename in filenames:
            img_path = os.path.join(dir_path, filename)
            # Convert the image to match preprocessing steps of training data
            _, np_preds, top_pred = get_single_prediction(img_path, model, img_size=img_size, grayscale=grayscale)

            label = list(dataset_dict['classes'].keys())[list(dataset_dict['classes'].values()).index(int(dirr))]
            if len(dataset_dict['classes']) == 2:
                binary_pred = 0 if top_pred in [0, 1] else 1
                misclassified = 1 if binary_pred != int(dirr) else 0
            else:
                misclassified = 1 if top_pred != int(dirr) else 0
            certain = 1 if (not misclassified) and np_preds.max() > 0.9 else 0
            undecided = 1 if np_preds.max() < 0.6 else 0
            correct = 1 if not misclassified else 0
            
            rows_all.append([filename, label, top_pred, *np_preds, correct, certain, misclassified, undecided])
            if misclassified:
                rows_misclassified.append([filename, label, top_pred, *np_preds])
            if certain:
                rows_certain.append([filename, label, top_pred, *np_preds])
            if undecided:
                rows_undecided.append([filename, label, top_pred, *np_preds])
            if correct:
                rows_correct.append([filename, label, top_pred, *np_preds])


    csv_path = os.path.join(csv_path, 'prediction_csvs')
    if not os.path.isdir(csv_path):
        os.mkdir(csv_path)
    df_all = pd.DataFrame(rows_all, columns=columns_all)
    df_all.to_csv(os.path.join(csv_path, 'all.csv'))

    df_correct = pd.DataFrame(rows_misclassified, columns=columns)
    df_correct.to_csv(os.path.join(csv_path,'correct.csv'))
    df_misclassified = pd.DataFrame(rows_misclassified, columns=columns)
    df_misclassified.to_csv(os.path.join(csv_path,'misclassified.csv'))
    df_certain = pd.DataFrame(rows_certain, columns=columns)
    df_certain.to_csv(os.path.join(csv_path,'certain.csv'))
    df_undecided = pd.DataFrame(rows_undecided, columns=columns)
    df_undecided.to_csv(os.path.join(csv_path,'undecided.csv'))

if __name__ == '__main__':
    pass
