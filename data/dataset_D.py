#%%
from omegaconf import omegaconf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from data import tokenizer_D as tokenizer

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

class CustomDataset_D(torch.utils.data.Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.discriminator_tokenizer = tokenizer.Tokenizer(config, model='discriminator')
        
        self.load_dataset()
    
    def load_dataset(self):
        self.orig_corpus = load_multi_txt(self.config.data[f'{self.mode}_orig_txt'])
        self.corrupt_corpus = load_multi_txt(self.config.data[f'{self.mode}_corrupt_txt'])
    
    def __len__(self):
        return len(self.orig_corpus)
    
    def __getitem__(self, idx):
        orig_line = self.orig_corpus[idx]
        corrupt_line = self.corrupt_corpus[idx]

        (input_token_ids, input_attn_mask, input_token_type_ids, target_ids) = \
                                self.discriminator_tokenizer.get_inp_feature(corrupt_line, orig_line)

        result = {}
        result['input_token_ids'] = input_token_ids
        result['input_attention_mask'] = input_attn_mask
        result['input_token_type_ids'] = input_token_type_ids
        result['target_ids'] = target_ids
        result['input_text'] = corrupt_line
        result['target_text'] = orig_line
        
        return result

if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("./config.yaml")

    train_dataset = CustomDataset_D(config)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=32)
    batch = next(iter(train_dataloader))
    print(batch)
    