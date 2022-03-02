#%%
import torch
import torch.nn as nn

from transformers import ElectraModel, ElectraTokenizer

#%%
class ELECTRA(nn.Module):
    def __init__(self, config):
        self.config = config
        self.model = ElectraModel.from_pretrained(config.pretrained_model)