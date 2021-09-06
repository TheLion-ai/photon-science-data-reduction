from sys import prefix
import numpy as np
from PIL import Image

from torch import load
from torch.nn import Linear, CrossEntropyLoss, BCELoss
from torch.optim import Adam
from torchvision.models.alexnet import AlexNet
from torchvision import models
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MetricCollection, Precision, Recall

from model_utils import load_trained_alexnet

class DiffractionModel(LightningModule):
    def __init__(self, backbone=AlexNet, encoder='imagenet', num_classes=7, tune_fc_only=True,  optimizer=Adam,
                lr=1e-3, loss=BCELoss):
        super().__init__()
        self.__dict__.update(locals())
        # init a pretrained resnet
        if encoder == 'imagenet':
            self.model = backbone(pretrained=True)
        else:
            self.model = load_trained_alexnet(encoder, num_classes)

        self.loss = loss()
        metrics = MetricCollection([Accuracy(), Precision(), Recall()])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        # Freeze feature extracting layers
        if tune_fc_only:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # Adjust the classifier to the no. of classes
        self.model.classifier[-1] = Linear(self.model.classifier[-1].in_features, num_classes)

    def forward(self, x):
       return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)

        metrics = self.train_metrics(preds, y)
        metrics = {**metrics, 'loss': loss}
        self.log_dict(metrics, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        metrics = self.val_metrics(preds, y)
        metrics = {**metrics, 'val_loss': loss}
        self.log_dict(metrics, on_epoch=True) 
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        metrics = self.test_metrics(preds, y)
        metrics = {**metrics, 'test_loss': loss}

