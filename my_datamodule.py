import pytorch_lightning as pl
import torch
from my_dataset import PMDataset


class PMDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        train_dataset = PMDataset(self.cfg, split='train')
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.dataloader.batch_size,
            num_workers=self.cfg.dataloader.num_workers,
            pin_memory=True,
            shuffle=self.cfg.dataloader.shuffle,
            persistent_workers=True
        )

    def val_dataloader(self):
        val_dataset = PMDataset(self.cfg, split='test')
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=self.cfg.dataloader.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        test_dataset = PMDataset(self.cfg, split='test')
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=self.cfg.dataloader.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True
        )
