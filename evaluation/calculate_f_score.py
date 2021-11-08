import os

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from utils.model_utils import get_single_prediction, load_trained_model
            
def calculate_f_score(model_dict, dataset_dict):

    y_true = []
    y_pred = []

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
            # Add new samples to list
            y_true.append(label)
            y_pred.append(top_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(f1_score(y_true, y_pred, average='weighted', labels=[0, 1, 2, 3, 4]))
    print(accuracy_score(y_true, y_pred))
    # return f1_score(y_true, y_pred, average='weighted')