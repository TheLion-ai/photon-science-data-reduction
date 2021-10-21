import os

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dicts.datasets_dict import datasets_dict
from dicts.pretrained_models_dict import pretrained_models_dict
from utils.model_utils import load_trained, get_single_prediction, load_trained_model
from utils.basic_utils import prepare_path
from config import config


def create_confusion_matrix(model_dict, dataset_dict):
# Creates a confusion matrix for the test set given the dataset and the model
    confusion_matrix = np.zeros((dataset_dict['num_classes'], model_dict['num_classes']))

    model = load_trained_model(model_dict['backbone'], model_dict['path'], model_dict['num_classes'], model_dict['img_size'])
    
    img_root_path = os.path.join(dataset_dict['path'], 'test')

    for dir in os.listdir(img_root_path):
        dir_path = os.path.join(img_root_path, dir)
        filenames = os.listdir(dir_path)
        
        for filename in filenames:
            img_path = os.path.join(dir_path, filename)
            # Get top prediction for a given image
            _, _, top_pred = get_single_prediction(img_path, model)
            # Get true label from dir name
            label = int(dir)
            # Add example to the desired place in the matrix
            confusion_matrix[label, top_pred] += 1
            
    plt.figure(figsize=(15,10))
    # Get class names to label confusion matrix rows and columns
    # dataset class names for the axis with true labels
    dataset_class_names = sorted(dataset_dict['classes'], key=dataset_dict['classes'].get)
    
    # model class names for the axis with true labels
    model_class_names = sorted(model_dict['classes'], key=model_dict['classes'].get)
    
    # Convert numpy array to confusion matrix with class names
    df_cm = pd.DataFrame(confusion_matrix, index=dataset_class_names, columns=model_class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    # Add row-column annotations
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fig_name = f'conf_matrix_{model_dict["name"]}_{dataset_dict["name"]}'
    path = prepare_path(model_dict['name'], dataset_dict['name'])
    path = os.path.join(path, fig_name)
    plt.savefig(path)


if __name__ == '__main__':
    
    model = '224_lenet_from_scratch'
    root_csv_path = f'prediction_csvs/{model}'

    for dataset in datasets_dict['diffranet']:
        dataset = datasets_dict['diffranet'][dataset]
        create_confusion_matrix(pretrained_models_dict[model], dataset)

