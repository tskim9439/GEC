#%%
import torch
import torch.nn as nn
import torch.utils.data
import pytorch_lightning as pl
import torchinfo

import numpy as np

from data import tokenizer_D, dataset_D
from models import discriminator as Model

from omegaconf import OmegaConf
config = OmegaConf.load("./config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
# Import dataset & dataloader
train_dataset = dataset_D.CustomDataset_D(config=config, mode='train')
val_dataset = dataset_D.CustomDataset_D(config=config, mode='val')

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            shuffle=True,
                                            batch_size=config.train_config.batch_size,
                                            num_workers=config.train_config.n_workers)

val_dataloder = torch.utils.data.DataLoader(dataset=val_dataset,
                                            shuffle=False,
                                            batch_size=config.train_config.batch_size,
                                            num_workers=config.train_config.n_workers)
#%%
# define callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
callbacks = [EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max"),
            ModelCheckpoint(
                        monitor="val_loss",
                        dirpath=config.train_config.root_dir ,#"my/path/",
                        filename="best_loss_{epoch:02d}_{val_loss:.3f}",
                        save_top_k=3,
                        mode="min",
                    ),
            ModelCheckpoint(
                        monitor="val_accuracy",
                        dirpath=config.train_config.root_dir ,#"my/path/",
                        filename="best_accuracy_{epoch:02d}_{val_accuracy:.3f}",
                        save_top_k=3,
                        mode="max",
                    )]
# Import model
model_D = Model.Discriminator(config)
# Model Summary
inp_ids = torch.ones((1, 128), dtype=torch.long)
inp_attn_mask = torch.ones((1, 128), dtype=torch.long)
inp_token_type = torch.ones((1, 128), dtype=torch.long)
torchinfo.summary(model_D,input_data=(inp_ids, inp_attn_mask, inp_token_type))

from trainer_D import DiscriminatorTrainer
model = DiscriminatorTrainer(config=config, model=model_D)
trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                    callbacks=callbacks,
                    default_root_dir="./logs")
trainer.fit(model=model,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloder)
