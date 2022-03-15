#%%
import torch
import torch.nn as nn

from transformers import AutoModelForPreTraining
#tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-generator")
#model = AutoModelForMaskedLM.from_pretrained("monologg/koelectra-base-v3-generator")


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.discriminator_config = config.model_config.discriminator

        # ELECTRA 의 Discriminator 는 이미 Binary Classifier 임
        self.model = AutoModelForPreTraining.from_pretrained(self.discriminator_config.pretrained_model)
        #self.model = AutoModel.from_pretrained(self.generator_config.pretrained_model)

    def forward(self, input_ids, attention_mask, token_type_ids):
        logit = self.model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids).logits
        return torch.sigmoid(logit)

    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == '__main__':
    from omegaconf import OmegaConf
    config = OmegaConf.load("/home/taesoo/GEC2/config.yaml")
    print(config)
    discriminator = Discriminator(config)

    inp_ids = torch.ones((1, 128), dtype=torch.long)
    inp_attn_mask = torch.ones((1, 128), dtype=torch.long)
    inp_token_type = torch.ones((1, 128), dtype=torch.long)

    output = discriminator(input_ids=inp_ids,
                    attention_mask=inp_attn_mask,
                    token_type_ids=inp_token_type)
                    #labels=inp_ids,
                    #output_attentions=inp_attn_mask)
    #loss = generator.model.compute_loss()
    print(output.shape) # [1, 128]
# %%
