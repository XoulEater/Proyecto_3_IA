import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

class DoubleConv(nn.Module):
    """(Conv => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetAutoencoder(pl.LightningModule):
    def __init__(self, in_channels=1, base_channels=32, latent_dim=128, learning_rate=1e-3):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck (latent)
        self.bottleneck = nn.Sequential(
            DoubleConv(base_channels * 2, base_channels * 4),
            nn.Conv2d(base_channels * 4, latent_dim, kernel_size=1)
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(latent_dim, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Output
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()  # Assuming reconstruction loss for autoencoder
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=10)
        self.auROC = torchmetrics.AUROC(task='multiclass', num_classes=10)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        x4 = self.pool2(x3)

        # Bottleneck
        x5 = self.bottleneck(x4)

        # Decoder with skip connections
        x6 = self.up2(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x7 = self.dec2(x6)
        x8 = self.up1(x7)
        x8 = torch.cat([x8, x1], dim=1)
        x9 = self.dec1(x8)

        return self.final_conv(x9)

    def _common_step(self, batch, batch_idx):
        #Get the samples and labels from the batch and apply the reshape 
        x, y = batch 
        x = x.reshape(x.size(0), -1)
        #Compute the forward step and get the scores
        scores = self.forward(x)
        #Apply the loss function
        loss = self.loss_fn(scores, y)
        #Return the loss, the scores, and the labels
        return loss, scores, y
    
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        auROC = self.auROC(scores, y)
        self.log_dict({'train_loss':loss, 'train_accuracy':accuracy, 'train_f1_score':f1_score, 'train_auROC':auROC}, 
                      on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, "scores": scores, "y": y}
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        auROC = self.auROC(scores, y)
        self.log_dict({'val_loss':loss, 'val_accuracy':accuracy, 'val_f1_score':f1_score, 'val_auROC':auROC}, 
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        auROC = self.auROC(scores, y)
        self.log_dict({'test_loss':loss, 'test_accuracy':accuracy, 'test_f1_score':f1_score, 'test_auROC':auROC}, 
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch 
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_encoder(self):
        """Retorna un módulo secuencial del encoder completo (útil para clasificadores)"""
        return nn.Sequential(
            self.enc1,
            self.pool1,
            self.enc2,
            self.pool2,
            self.bottleneck
        )
