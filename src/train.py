from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl


# Logger con Weights & Biases
def train(
    model,
    train_dataloaders,
    datamodule,
    project_name,
    run_name,
    max_epochs,
    early_stop_patience,
    checkpoint_dir,
):
    wandb_logger = WandbLogger(project=project_name, log_model=True)
    wandb_logger.experiment.name = run_name
    wandb_logger.experiment.save()

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=early_stop_patience,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.8f}",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=3
    )

    trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=datamodule.val_dataloader())
    trainer.test(model, dataloaders=datamodule.test_dataloader())
    wandb_logger.experiment.finish()

# Example usage:
# if __name__ == "__main__":
#     from dataset import SemiSupervisedMnistDataModule
#     from unet import UNet
#     from classifier import Classifier
#     import os
#     data_dir = "path/to/data"
#     batch_size = 32
#     dm = SemiSupervisedMnistDataModule(data_dir, batch_size)
#     model = UNet()
#     classifier = Classifier()
#     train_autoencoder(
#         model=model,
#         datamodule=dm,
#         project_name="autoencoder_project",
#         run_name="autoencoder_run",
#         max_epochs=50,
#         early_stop_patience=5,
#         checkpoint_dir=os.path.join("checkpoints", "autoencoder")
#     )
