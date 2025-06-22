import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

class ClassifierA(pl.LightningModule):
    def __init__(self, encoder: nn.Module, latent_dim=128, num_classes=30, learning_rate=1e-3):
        super().__init__()

        self.encoder = encoder  # Encoder no preentrenado
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Para reducir a vector
            nn.Flatten(),
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
