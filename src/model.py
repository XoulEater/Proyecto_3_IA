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
    def __init__(self, in_channels=3, base_channels=64, latent_dim=128, learning_rate=1e-3):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels)            # 128x128 -> 64x64
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)      # 64x64 -> 32x32
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DoubleConv(base_channels * 2, base_channels * 4),         # 32x32
            nn.Conv2d(base_channels * 4, latent_dim, kernel_size=1)  # 32x32x128
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(latent_dim, base_channels * 2, kernel_size=2, stride=2)  # 32x32 -> 64x64
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)  # 64x64 -> 128x128
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Output
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x1 = self.enc1(x)          # 128x128 -> 128x128
        x2 = self.pool1(x1)        # 128x128 -> 64x64
        x3 = self.enc2(x2)         # 64x64
        x4 = self.pool2(x3)        # 64x64 -> 32x32
        x5 = self.bottleneck(x4)   # 32x32

        x6 = self.up2(x5)          # 32x32 -> 64x64
        x6 = torch.cat([x6, x3], dim=1)
        x7 = self.dec2(x6)

        x8 = self.up1(x7)          # 64x64 -> 128x128
        x8 = torch.cat([x8, x1], dim=1)
        x9 = self.dec1(x8)

        return self.final_conv(x9)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_fn(x_hat, x)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_fn(x_hat, x)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_encoder(self):
        return nn.Sequential(
            self.enc1,
            self.pool1,
            self.enc2,
            self.pool2,
            self.bottleneck
        )