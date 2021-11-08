from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from model.diffraction_model import DiffractionModel
from utils.data_utils import get_dataloaders


def train(dataloader_params, model_params, trainer_params, model_id):
    """A function for model training.

    Args:
        dataloader_params (dict): A dict with all parameters required by the dataloader.
        model_params (dict): A dict with all parameters required by the model that is going to be trained.
        trainer_params (dict): A dict with all parameters required by the model trainer.
        model_id ('str'): The id of the model that is going to be created.
    """
    dataloaders = get_dataloaders(**dataloader_params)
    model = DiffractionModel(**model_params)

    logger = CSVLogger(save_dir="logs/", name=model_id)
    
    checkpoint_callback = ModelCheckpoint(
                            monitor="val_loss",
                            dirpath=f"logs/{model_id}",
                            filename="-{epoch:02d}-{val_loss:.2f}-{accuracy:.2f}",
                            save_top_k=1,
                            mode="min",
                        )
    early_stopping = EarlyStopping('val_loss', patience=50)

    trainer = Trainer(**trainer_params, logger=logger, callbacks=[checkpoint_callback, early_stopping])
    trainer.fit(model, *dataloaders)

if __name__ == '__main__':
    pass
    # dataloader_params = config['dataloader']
    # model_params = config['model']
    # trainer_params = config['trainer']
    # model_id = config['model_id']

    # train(dataloader_params, model_params, trainer_params, model_id)