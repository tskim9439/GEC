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
test_dataset = dataset_D.CustomDataset_D(config=config, mode='test')
test_dataloder = torch.utils.data.DataLoader(dataset=test_dataset,
                                            shuffle=False,
                                            batch_size=config.train_config.batch_size,
                                            num_workers=config.train_config.n_workers)
# Import model
model_D = Model.Discriminator(config)
from trainer_D import DiscriminatorTrainer
from pytorch_lightning.callbacks import ModelCheckpoint
model = DiscriminatorTrainer(config=config, model=model_D)
model = model.load_from_checkpoint("/home/taesoo/GEC2/logs/best_accuracy_epoch=12_val_accuracy=0.862.ckpt", config=config, model=model_D)
model.eval()
model_D.load_state_dict(model.model.state_dict())
# %%
batch = next(iter(test_dataloder))
# %%
logits = model_D(batch["input_token_ids"], batch["input_attention_mask"], batch["input_token_type_ids"])
y_hat = torch.round(logits)
#%%%
y_hat[-2], batch["input_text"][-2]
# %%