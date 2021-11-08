import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.model_utils import get_single_prediction, load_trained_model
from utils.basic_utils import prepare_results_path
from sklearn.metrics import f1_score, accuracy_score
            

def create_confusion_matrix(model_dict, dataset_dict):
# Creates a confusion matrix for the test set given the dataset and the model
    y_true = []
    y_pred = []
    
    confusion_matrix = np.zeros((dataset_dict['num_classes'], model_dict['num_classes']))

    model = load_trained_model(model_dict['arch'], model_dict['path'], model_dict['num_classes'], model_dict['img_size'])
    
    img_root_path = os.path.join(dataset_dict['path'], dataset_dict['test_dir'])

    for dir in os.listdir(img_root_path):
        dir_path = os.path.join(img_root_path, dir)
        filenames = os.listdir(dir_path)
        
        for filename in filenames:
            img_path = os.path.join(dir_path, filename)
            # Get top prediction for a given image
            _, _, top_pred = get_single_prediction(img_path, model, model_dict['img_size'], model_dict['grayscale'])
            # Get true label from dir name
            label = int(dir)
            # Add example to the desired place in the matrix
            confusion_matrix[label, top_pred] += 1
            y_true.append(label)
            y_pred.append(top_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print('F1 score:', f1_score(y_true, y_pred, average='weighted', labels=[0, 1, 2, 3, 4]))
    print('Accuracy: ', accuracy_score(y_true, y_pred))
            
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
    path = prepare_results_path(model_dict, dataset_dict)
    path = os.path.join(path, fig_name)
    plt.savefig(path)


if __name__ == '__main__':
    pass
