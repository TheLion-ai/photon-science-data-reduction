""" A configuration file for model training.
"""
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam, SGD, lr_scheduler
from torchvision.models import alexnet, resnet50, vgg19

from dicts.datasets_dict import datasets_dict
from dicts.pretrained_models_dict import pretrained_models_dict

from model.train import train
from model.lenet import LeNet

root_dir = 'home/bk/sfx/model'
tune_fc_only = True
dataset = datasets_dict['ln84']['ln84_preprocessed']
backbone = vgg19
encoder = 'imagenet'
img_size = 224
loss = BCELoss if dataset['multilabel'] else CrossEntropyLoss

config = {
    'dataloader': {
        'dataset_dict': dataset,
        'batch_size': 64,
        'img_size': img_size,
        'grayscale': False
        },

    'model': {
            'backbone': backbone,
            # 'encoder_path': pretrained_models_dict[encoder]['path'],
            'num_classes': dataset['num_classes'],
            'img_size': img_size,
            'tune_fc_only': tune_fc_only,
            'optimizer': Adam,
            'loss': loss,
            'lr': 1e-4
        },

    'trainer': {
        'gpus': [0, 1, 2],
        'max_epochs': 1500
    },

    'model_id': f"{img_size}_{backbone.__name__}_tune_fc_only{tune_fc_only}_{dataset['name']}"
}

if __name__ == '__main__':

    # backbones = [alexnet, LeNet, vgg19]
    # for backbone in backbones:
    #     for tune_fc_only in [True, False]:
    #         dataloader_params = config['dataloader']
    #         model_params = config['model']
    #         trainer_params = config['trainer']
    #         model_id = config['model_id']
    #         model_params['backbone'] = backbone
    #         model_params['tune_fc_only'] = tune_fc_only
    #         if backbone.__name__ == 'alexnet':
    #             dataloader_params['img_size'] = 227
    #             model_params['img_size'] = 227
    #             model_params['encoder_path'] = pretrained_models_dict['alexnet_all_layers']['path']
    #         elif backbone.__name__ == 'vgg19':
    #             dataloader_params['img_size'] = 224
    #             model_params['img_size'] = 224
    #             model_params['encoder_path'] = pretrained_models_dict['vgg19']['path']
    #         elif backbone.__name__ == 'lenet':
    #             dataloader_params['img_size'] = 32
    #             model_params['img_size'] = 32
    #             model_params['encoder_path'] = pretrained_models_dict['lenet_32']['path']
    #             if tune_fc_only:
    #                 pass

    # encoder_path =  pretrained_models_dict['vgg19_all_layers']['path']
    dataloader_params = config['dataloader']
    model_params = config['model']
    trainer_params = config['trainer']
    model_id = config['model_id']
    train(dataloader_params, model_params, trainer_params, model_id)

