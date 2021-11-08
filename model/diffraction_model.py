from torch import load
from torch.nn import Linear, CrossEntropyLoss, BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import alexnet, resnet50, vgg19
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MetricCollection, Precision, Recall

from utils.model_utils import load_trained_model


class DiffractionModel(LightningModule):

    def __init__(self, arch=alexnet, encoder_path='', num_classes=5, img_size=224, tune_fc_only=True,  optimizer=Adam,
                lr=1e-3, loss=BCELoss):
        """A model for training on diffraction images.
        It can be built on one of the accepted architectures.


        Args:
            arch (class, optional): The architecture with which the model is built. Select from alexnet, vgg19, alexnet or LeNet. Defaults to alexnet.
            encoder_path (str, optional): A path to the pretrained model with labels used as . Defaults to {}.
            num_classes (int, optional): The number of classes that the model is going to predict. Defaults to 5.
            img_size (int, optional): Image size requires of the model input. Defaults to 224.
            tune_fc_only (bool, optional): If True, only feature connect layers are going to be trained. Else all layers will be trained. Defaults to True.
            optimizer (class methos, optional): An optimizer used by the model. Defaults to Adam.
            lr (float, optional): The learning rate used by the model. Defaults to 1e-3.
            loss (class method, optional): Loss function used by the model. Defaults to BCELoss.
        """
        super().__init__()
        self.__dict__.update(locals())
        self.model = load_trained_model(arch, encoder_path, num_classes, img_size)

        self.loss = loss()
        metrics = MetricCollection([Accuracy(), Precision(), Recall()])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

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
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-20,
            verbose=True
        )
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_loss'
       }

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
