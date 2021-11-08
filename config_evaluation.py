import os

from dicts.datasets_dict import datasets_dict
from dicts.pretrained_models_dict import pretrained_models_dict
from utils.basic_utils import prepare_results_path
from evaluation.prediction_csvs import get_prediction_csvs
from evaluation.confusion_matrix import create_confusion_matrix
from evaluation.calculate_f_score import calculate_f_score

# Create confusion matrix
model_dict = pretrained_models_dict['vgg19-fc-ln84_preprocessed']
dataset_dict = datasets_dict['ln84_preprocessed']

# TODO: csv path tworzony wewnętrznie w funkcji
csv_path = prepare_results_path(model_dict, dataset_dict)

# create_confusion_matrix(model_dict, dataset_dict)
get_prediction_csvs(model_dict, dataset_dict, csv_path)
# for dataset in datasets_dict['diffranet']:
#     dataset = datasets_dict['diffranet'][dataset]

#     # create_confusion_matrix(pretrained_models_dict[model], dataset)
#     break


# for model_name in ['vgg19_all_layers', 'vgg19_fc']:
#     model_dict = pretrained_models_dict[model_name]
#     img_size = model_dict['img_size']
#     backbone = model_dict['backbone']
#     model_path = model_dict['path']
#     num_classes = model_dict['num_classes']

#     for dataset in datasets_dict['diffranet'].keys():
#         dataset_dict = datasets_dict['diffranet'][dataset]
#         csv_path = prepare_path(model_dict['name'], dataset_dict['name'])
#         # os.path.join(csv_root_dir, dataset['name'])

#         if not os.path.isdir(csv_path):
#             os.mkdir(csv_path)
#         # get_prediction_csvs(model_dict, dataset_dict, csv_path)
