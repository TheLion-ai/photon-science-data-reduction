from datetime import datetime

from utils.data_utils import get_dataloaders
from utils.model_utils import load_trained_model



from dicts.pretrained_models_dict import pretrained_models_dict


from config import config
# TODO: ADD img size

model = '224_lenet_from_scratch'
model_path = pretrained_models_dict[model]['path']
dataloaders = get_dataloaders(**config['dataloader'])
model = load_trained_model(pretrained_models_dict[model]['backbone'], model_path, 5, pretrained_models_dict[model]['img_size'], pretrained_models_dict[model]['grayscale'])

print(config['dataloader']['batch_size'])
start=datetime.now()
a = model(next(iter(dataloaders[0]))[0])
print(datetime.now()-start)
