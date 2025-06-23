import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

class QuantizedClassifier(pl.LightningModule):
    def __init__(self, encoder: nn.Module, num_classes=30, learning_rate=1e-3, freeze_encoder=False):
        super().__init__()

        self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.feature_size = 128 * 32 * 32  # latent_dim * height * width

        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # Cabeza de clasificación
        self.classifier = nn.Sequential(
            # Global Average Pooling para reducir dimensionalidad
            nn.AdaptiveAvgPool2d((4, 4)),  # Reduce 32x32 -> 4x4
            nn.Flatten(),
            
            # Capas completamente conectadas
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, num_classes)
        )

        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)
        self.auROC = torchmetrics.AUROC(task='multiclass', num_classes=num_classes)


    def forward(self, x):
        x = self.quant(x)
        with torch.set_grad_enabled(self.encoder.training):
            features = self.encoder(x)  # Shape: [batch_size, 128, 32, 32]

        logits = self.classifier(features)
        logits = self.dequant(logits)
        return logits
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
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
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
