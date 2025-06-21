import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics


class ClassifierFrozen(pl.LightningModule):
    def __init__(self, pretrained_encoder, num_classes=10, learning_rate=1e-3):
        super().__init__()
        self.encoder = pretrained_encoder
        for param in self.encoder.parameters():
            param.requires_grad = False  # congelar encoder
        self.classifier = nn.Linear(128, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
