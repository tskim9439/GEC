#%%
from omegaconf import omegaconf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from data import tokenizer

def load_txt_file(filepath):
    f = open(filepath, "r")
    lines = f.readlines()
    lines = list(map(lambda x : x.strip(), lines))
    f.close()
    return lines

def load_multi_txt(filepaths):
    lines = []
    for filepath in filepaths:
        lines.extend(load_txt_file(filepath))
    return lines

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.critic_tokenizer = tokenizer.Tokenizer(config, model='critic')
        self.generator_tokenizer = tokenizer.Tokenizer(config, model='generator')
        
        self.load_dataset()
    
    def load_dataset(self):
        self.orig_corpus = load_multi_txt(self.config.data[f'{self.mode}_orig_txt'])
        self.corrupt_corpus = load_multi_txt(self.config.data[f'{self.mode}_corrupt_txt'])
    
    def __len__(self):
        return len(self.orig_corpus)
    
    def __getitem__(self, idx):
        orig_line = self.orig_corpus[idx]
        corrupt_line = self.corrupt_corpus[idx]

        orig_token_ids, orig_attn_mask, orig_token_type_ids = self.generator_tokenizer.get_inp_feature(orig_line)
        corrupt_token_ids, corrupt_attn_mask, corrupt_token_type_ids = self.generator_tokenizer.get_inp_feature(corrupt_line)

        result = {}
        result['orig_token_ids'] = orig_token_ids
        result['orig_attn_mask'] = orig_attn_mask
        result['orig_token_type_ids'] = orig_token_type_ids
        result['corrupt_token_ids'] = corrupt_token_ids
        result['corrupt_attn_mask'] = corrupt_attn_mask
        result['corrupt_token_type_ids'] = corrupt_token_type_ids
        
        return result

if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("./config.yaml")

    train_dataset = CustomDataset(config)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=32)
    batch = next(iter(train_dataloader))
    print(batch)
    