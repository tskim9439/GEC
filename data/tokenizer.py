#%%
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from glob import glob

class Tokenizer:
    def __init__(self, config, model='critic'):
        self.config = config
        self.pretrained_model = config.model_config[model]['pretrained_model']
        self.tokenizer = self.get_tokenizer(self.pretrained_model)
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.special_tokens = self.tokenizer.all_special_tokens
        self.max_len = self.config.model_config.max_len

    def get_tokenizer(self, pretrained_model_path):
        return AutoTokenizer.from_pretrained(pretrained_model_path)
    
    def tokenize(self, line):
        tokens = []
        words = line.split(' ')
        for word in words:
            w_token = self.tokenizer.tokenize(word)
            n_pad_token = len(word) - len(w_token)
            pad = [self.pad_token * n_pad_token]
            if n_pad_token > 0:
                w_token.extend(pad)
            tokens.extend(w_token)
        return tokens
    
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)
    
    def convert_tokens_to_string(self, tokens):
        string = self.tokenizer.convert_tokens_to_string(tokens)
        string = string.replace(' [PAD]', '')
        return string
    
    def get_inp_feature(self, string):
        # Tokenize
        token = self.tokenize(string)
        token_ids = self.convert_tokens_to_ids(token)
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        
        attn_mask = [1] * len(token_ids)
        token_type_ids = [0] * len(token_ids)

        # Padding
        pad = [self.pad_token_id] * int(self.max_len - len(token_ids))
        token_ids = token_ids + pad
        attn_mask = attn_mask + pad
        token_type_ids = token_type_ids + pad

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        attn_mask = torch.tensor(attn_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        return token_ids, attn_mask, token_type_ids

def get_tokenizer(pretrained_model_path):
        return AutoTokenizer.from_pretrained(pretrained_model_path)

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

if __name__ == "__main__":
    # 음절보다 많은 수의 Token으로 Tokenize 하는 경우는 없다.
    from omegaconf import OmegaConf
    config = OmegaConf.load("./config.yaml")

    # Load txt files
    orig_txt_files = glob(os.path.join(config.data.root_dir, "orig*.txt"))
    orig_txt_files.sort()
    corrupt_txt_files = glob(os.path.join(config.data.root_dir, "corr*.txt"))
    corrupt_txt_files.sort()
    print("Original txt files")
    print(orig_txt_files)
    print("Corrupted txt files")
    print(corrupt_txt_files)
    orig_lines = load_multi_txt(orig_txt_files)
    corrupt_lines = load_multi_txt(corrupt_txt_files)
    print(f'number of original lines {len(orig_lines)}')
    print(f'number of corrupted lines {len(corrupt_lines)}')
    print(orig_lines[1], len(orig_lines[1]))
    print(corrupt_lines[1], len(corrupt_lines[1]))

    tokenizer = Tokenizer(config)
    #critic_tokenizer = get_tokenizer(config.model_config.critic.pretrained_model)
    #generator_tokenizer = get_tokenizer(config.model_config.generator.pretrained_model)

    #orig_token = critic_tokenizer.tokenize(orig_lines[1])
    #corrupt_token = critic_tokenizer.tokenize(corrupt_lines[1])

    print(tokenizer.tokenize(orig_lines[1]))
    print(tokenizer.tokenize(corrupt_lines[1]))
    #print(corrupt_token)
    line = orig_lines[1]
    token = tokenizer.tokenize(line)
    ids = tokenizer.convert_tokens_to_ids(token)
    print(f'Ids : {ids}')
    back_token = tokenizer.convert_ids_to_tokens(ids)
    print(f'back tokens {back_token}')
    back_string = tokenizer.convert_tokens_to_string(back_token)
    print(f'back string {back_string}')
    print(tokenizer.special_tokens)

    inp_ids, inp_attn_mask, inp_type_ids = tokenizer.get_inp_feature(corrupt_lines[1])
    out_ids, out_attn_mask, out_type_ids = tokenizer.get_inp_feature(orig_lines[1])
# %%
