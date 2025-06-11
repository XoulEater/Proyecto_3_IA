from typing import Optional
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_lightning as pl
import numpy as np
import os

class SemiSupervisedMnistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 4,
        label_pct: float = 0.3,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_pct = label_pct
        self.seed = seed

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
        ])

    def setup(self, stage: Optional[str] = None):
        full_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'train'),
            transform=self.transform
        )

        # Shuffle indices
        total_size = len(full_dataset)
        indices = np.arange(total_size)
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        # Split indices
        label_count = int(total_size * self.label_pct)
        labeled_indices = indices[:label_count]
        unlabeled_indices = indices[label_count:]

        self.labeled_dataset = Subset(full_dataset, labeled_indices)
        self.unlabeled_dataset = Subset(full_dataset, unlabeled_indices)

        # Test y val directamente
        self.val_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'valid'),
            transform=self.transform
        )

        self.test_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'test'),
            transform=self.transform
        )

    def labeled_dataloader(self):
        return DataLoader(
            self.labeled_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def unlabeled_dataloader(self):
        return DataLoader(
            self.unlabeled_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
