import argparse
import os
from datetime import datetime

import omegaconf
import pytorch_lightning as pl
import torch
from my_datamodule import PMDataModule
from my_dataset import PMDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from urdformer import URDFormer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='my_cfg.yaml')
    args = parser.parse_args()

    cfg = omegaconf.OmegaConf.load(args.cfg)
    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    pl.seed_everything(cfg.train.seed)

    dm = PMDataModule(cfg)

    model = URDFormer(**cfg.URDFormer)
    if cfg.train.resume:
        print(f"Resuming training from checkpoint {cfg.train.resume_path}")
        model.load_state_dict(torch.load(cfg.train.resume_path)['model_state_dict'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_dir = os.path.join(f'{cfg.train.save_dir}/{cfg.logger.name}', f'{timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Checkpoint directory: {checkpoint_dir}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=cfg.train.save_interval,
        save_last=True,)

    logger = WandbLogger(**cfg.logger)
    logger.log_hyperparams(cfg_dict)
    print(f"WandbLogger initialized with project: {logger.experiment.project}, name: {logger.experiment.name}")

    trainer = pl.Trainer(max_epochs=cfg.train.epochs,
                         logger=logger,
                         gpus=1,
                         accelerator='gpu',
                         strategy='ddp',
                         check_val_every_n_epoch=cfg.train.val_interval,
                         log_every_n_steps=1,
                         callbacks=[checkpoint_callback],
                         gradient_clip_val=2.0,
                         gradient_clip_algorithm="norm",)
    trainer.fit(model, datamodule=dm)

    print(f"Checkpoints saved at: {checkpoint_dir}")
    print('done')
