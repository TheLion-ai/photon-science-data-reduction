import os

import pandas as pd

from utils.model_utils import load_trained_model, get_single_prediction

# TODO: folder z błędami LN84
# TODO: liczebność LN84


def get_prediction_csvs(model_dict, dataset_dict, csv_path):
# Creates a separate csv for images from categories: correct, certain, misclassified and undecided
# and a joint csv for all of the above
    model_path = model_dict['path']
    arch = model_dict['arch']
    num_classes = model_dict['num_classes']
    img_size = model_dict['img_size']
    grayscale = model_dict['grayscale']
    pred_qualities = {'correct': 0, 'certain': 0, 'misclassified': 0, 'undecided': 0}
    pred_qualities_names = list(pred_qualities.keys())

    model = load_trained_model(arch, model_path, num_classes, img_size, fine_tune=True)
    root_path = os.path.join(dataset_dict['path'], dataset_dict['test_dir'])

    columns = ['id', 'label', 'top_pred'] + list(dataset_dict['classes'].values())
    columns_all = columns + pred_qualities_names
    rows = {k: [] for k in ['all', *pred_qualities_names]}

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
            if model_dict['num_classes'] == 5 and dataset_dict['num_classes'] == 2:
                binary_pred = 0 if top_pred in [0, 1] else 1
                pred_qualities['misclassified'] = 1 if binary_pred != int(dirr) else 0
            else:
                pred_qualities['misclassified'] = 1 if top_pred != int(dirr) else 0
            pred_qualities['certain'] = 1 if (not pred_qualities['misclassified']) and np_preds.max() > 0.9 else 0
            pred_qualities['undecided'] = 1 if np_preds.max() < 0.6 else 0
            pred_qualities['correct'] = 0 if pred_qualities['misclassified'] else 1
            
            rows['all'].append([filename, label, top_pred, *np_preds, *pred_qualities_names])

            for name in pred_qualities_names:
                if pred_qualities[name]:
                    rows[name].append([filename, label, top_pred, *np_preds])

    csv_path = os.path.join(csv_path, 'prediction_csvs')
    if not os.path.isdir(csv_path):
        os.mkdir(csv_path)


    df_all = pd.DataFrame(rows['all'], columns=columns_all)
    df_all.to_csv(os.path.join(csv_path, 'all.csv'))

    for name in pred_qualities_names:
        df = pd.DataFrame(rows[name], columns=columns)
        df.to_csv(os.path.join(csv_path, f"{name}.csv"))

if __name__ == '__main__':
    pass
