from torch import load
from torch.nn import Linear
from torchvision.models.alexnet import AlexNet

def load_trained_alexnet(model_path, num_classes):
    model = AlexNet()
    model.classifier[-1] = Linear(model.classifier[-1].in_features, num_classes)
    state_dict = load(model_path)['state_dict']
    state_dict = {k.partition('model.')[2]: v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model