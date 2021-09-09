import os

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import load
from torch.nn import Linear  
from torch.nn.functional import softmax  
from torchvision.models.alexnet import AlexNet

from datasets_dict import datasets_dict
from data_utils import get_transforms
from model_utils import load_trained_alexnet
from config import config, pretrained_models

def get_single_prediction(img_path, model):
    img = Image.open(img_path).convert('RGB')
    my_transforms = get_transforms()
    x = my_transforms(img)
    preds = model(x.unsqueeze(0))
    preds = softmax(preds[0], dim=0)
    np_preds = preds.detach().numpy()
    top_pred = list(np_preds).index(np_preds.max())

    return preds, np_preds, top_pred


def get_prediction_csvs(model_path, dataset, csv_path):
# Creates a separate csv for maximum 10 images from categories: certain, misclassified and undecided
# and a joint csv for all of the above
    num_classes = 5
    confusion_matrix = np.zeros((num_classes, num_classes))
    model = load_trained_alexnet(model_path, 5)
    root_path = os.path.join(dataset['path'], 'test')

    columns = ['id', 'label', 'top_pred'] + list(dataset['classes'].values())
    columns_all = columns + ['certain', 'misclassified', 'undecided']
    rows_all = []
    rows_certain = []
    rows_misclassified = []
    rows_undecided = []

    for dir in os.listdir(root_path):
        dir_path = os.path.join(root_path, dir)
        filenames = os.listdir(dir_path)
        
        for filename in filenames:
            img_path = os.path.join(dir_path, filename)
            # Convert the image to match preprocessing steps of training data
            preds, np_preds, top_pred = get_single_prediction(img_path, model)

            label = dataset['classes'][int(dir)]
            if dataset['num_classes'] == 2:
                binary_pred = 0 if top_pred in [0, 1] else 1
                misclassified = 1 if binary_pred != int(dir) else 0
            else:
                misclassified = 1 if top_pred != int(dir) else 0
                confusion_matrix[int(dir), top_pred] += 1
            certain = 1 if (not misclassified) and np_preds.max() > 0.9 else 0
            undecided = 1 if np_preds.max() < 0.6 else 0
            
            rows_all.append([filename, label, top_pred, *np_preds, certain, misclassified, undecided])
            if misclassified:
                rows_misclassified.append([filename, label, top_pred, *np_preds])
            if certain:
                rows_certain.append([filename, label, top_pred, *np_preds])
            if undecided:
                rows_undecided.append([filename, label, top_pred, *np_preds])

    df_all = pd.DataFrame(rows_all, columns=columns_all)
    df_all.to_csv(os.path.join(csv_path, 'all_really.csv'))

    plt.figure(figsize=(15,10))

    class_names = ['blank', 'no-crystal', 'weak', 'good', 'strong']
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'conf_matrix_{dataset["name"]}')


if __name__ == '__main__':
    
    csv_root_dir = 'prediction_csvs/alexnet_fine_tune_imagenet'
    model = 'imagenet_finetune'

    for dataset in datasets_dict['diffranet']:
        dataset = datasets_dict['diffranet'][dataset]
        csv_path = os.path.join(csv_root_dir, dataset['name'])
        if not os.path.isdir(csv_path):
            os.mkdir(csv_path)
        get_prediction_csvs(pretrained_models[model]['path'], dataset, csv_path)

