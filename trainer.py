#%%
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as FM

import numpy as np
# %%
class Trainer(pl.LightningModule):
    def __init__(self, config, model):
        self.config = config
        self.model = model

        self.configure_loss_fn()
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        return self.model['generator'](input_ids, attn_mask, token_type_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch['corrupt_token_ids']
        attn_mask = batch['corrupt_attn_mask']
        token_type_ids = batch['corrupt_token_type_ids']

        logits = self(input_ids=input_ids,
                    attn_mask=attn_mask,
                    token_type_ids=token_type_ids) # model forward
        loss = self.maskNLLLoss(logits, batch['orig_token_ids'], batch['orig_attn_mask'])
        return loss

    def maskNLLLoss(self, inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        return loss / nTotal # mean

    def validation_step(self, batch):
        input_ids = batch['corrupt_token_ids']
        attn_mask = batch['corrupt_attn_mask']
        token_type_ids = batch['corrupt_token_type_ids']

        logits = self(input_ids=input_ids,
                    attn_mask=attn_mask,
                    token_type_ids=token_type_ids) # model forward
        loss = self.maskNLLLoss(logits, batch['orig_token_ids'], batch['orig_attn_mask'])
        acc = FM.accuracy(logits, batch['orig_token_ids'])
        metrics = {'val_acc':acc, 'val_loss':loss}
        self.log_dict(metrics)
        
    def test_step(self, batch):
        pass

    def configure_loss_fn(self):
        self.loss_fn = nn.ModuleDict({
            'generator_loss_fn' : nn.CrossEntropyLoss()
        })

    def configure_optimizers(self):
        optimizer = nn.ModuleDict({
            'generator_optimizer' : torch.Adam(self.parameters(), lr=self.config.train_config.lr),
        })
        return optimizer

#%%
