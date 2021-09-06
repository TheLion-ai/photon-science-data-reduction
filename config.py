from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam, lr_scheduler
from torchvision.models import alexnet

from datasets_dict import datasets_dict
from train import train

pretrained_models = {
    'imagenet': {},
    'diffranet_fulltune': {
        'path': '/home/bk/sfx/logs/alexnet_full_tune_imagenet-epoch=12-val_loss=0.18.ckpt',
        'backbone': alexnet,
        'num_classes': 5
    },
    'diffranet_finetune': {
        'path': '/home/bk/sfx/logs/alexnet_fine_tune_diffranet-epoch=22-val_loss=0.22.ckpt',
        'backbone': alexnet,
        'num_classes': 5
    }
}

root_dir = 'home/bk/sfx/model'
tune_fc_only = True
dataset = datasets_dict['diffranet']['synthetic']
backbone = alexnet
loss = BCELoss if dataset['multilabel'] else CrossEntropyLoss

config = {
    'dataloader': {
        'dataset': dataset,
        'batch_size': 1024
        },

    'model': {
            'backbone': backbone,
            'encoder': 'imagenet',
            'num_classes': dataset['num_classes'],
            'tune_fc_only': tune_fc_only,
            'optimizer': Adam,
            'loss': loss,
            'lr': 1e-3
        },

    'trainer': {
        'gpus': [0, 1, 2],
        'max_epochs': 50
    },

    'model_id': f'TESSST{backbone.__name__}_tune_fc_only{tune_fc_only}_{dataset}'
}

if __name__ == '__main__':
    dataloader_params = config['dataloader']
    model_params = config['model']
    trainer_params = config['trainer']
    model_id = config['model_id']

    train(dataloader_params, model_params, trainer_params, model_id)

