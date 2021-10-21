"""
The module contains a set of functions for using the pretrained models.
"""
from PIL import Image

from torch import load
from torch.nn import Linear, Module, Sequential, Conv2d, Tanh, AvgPool2d, Flatten, AdaptiveAvgPool2d
from torch.nn.functional import softmax
from torchvision.models import alexnet, resnet50, vgg19

from model.lenet import LeNet
from utils.data_utils import get_transforms

def load_trained_model(backbone, model_path, num_classes, img_size, fine_tune=True):
    """[summary]

    Args:
        backbone (class): The architecture the model was built with.
        model_path (str): The path to the pretrained model.
        num_classes (int): The number of classes distinguished by the model.
        img_size (int): Input image size required by the model.
        fine_tune (bool, optional): If True, only fc layers are going to be trained.. Defaults to True.

    Returns:
        class: A model with pretrained weights.
    """
    pretrained = False if model_path else True

    if backbone.__name__ == 'LeNet':
        model = LeNet(num_classes, img_size)
    else:
        model = backbone(pretrained=pretrained)

    model.classifier[-1] = Linear(model.classifier[-1].in_features, 5) # TODO: Remove hardcoding
    if model_path:
        state_dict = load(model_path)['state_dict'] if backbone.__name__ not in ["LeNet"] else load(model_path)
        state_dict = {k.partition('model.')[2]: v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)


    if backbone.__name__ in ['alexnet', 'vgg19', 'LeNet']:
        if fine_tune:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[-1] = Linear(model.classifier[-1].in_features, num_classes)
    elif backbone.__name__ in ['resnet50']:
        # ResNet has different naming of layers
        if fine_tune:
            for param in model.parameters():
                param.requires_grad = False
        # Replace the last layer with a layer with desired no. of classes
        model.fc = Linear(model.fc.in_features, num_classes)
        model.fc.requires_grad = True


    return model


def get_single_prediction(img_path, model, img_size=224, grayscale=False):
    """Get a prediction for a single image.

    Args:
        img_path (str): a path to the input image,
        model (class instance): the model that is used to get the prediction,
        img_size (int, optional): Input image size required by the model. Defaults to 224.

    Returns:
        torch layer: A torch layer with prediction.
        numpy array: A numpy array with the predictions.
        int: the index of the class with the highest predicted score.
    """
    img = Image.open(img_path).convert('RGB')
    my_transforms = get_transforms(img_size, grayscale)
    x = my_transforms(img)
    preds = model(x.unsqueeze(0))
    preds = softmax(preds[0], dim=0)
    np_preds = preds.detach().numpy()
    top_pred = list(np_preds).index(np_preds.max())

    return preds, np_preds, top_pred