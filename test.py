import os
from datetime import datetime

from torch.cuda import device_count

from utils.data_utils import get_dataloaders
from utils.model_utils import load_trained_model
from dicts.pretrained_models_dict import pretrained_models_dict


from config import config

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
print("Let's use", device_count(), "GPUs!")

model_dict = pretrained_models_dict['alexnet_fc']
dataloaders = get_dataloaders(**config['dataloader'])
model = load_trained_model(model_dict['arch'], model_dict['path'], model_dict['num_classes'], model_dict['img_size'], model_dict['grayscale'])

print(f"Tested model: {model_dict['name']} with bach_size: {config['dataloader']['batch_size']}")
start=datetime.now()

preds = model(next(iter(dataloaders[0]))[0])

print(datetime.now()-start)
