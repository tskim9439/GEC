#%%
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
#tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-generator")
#model = AutoModelForMaskedLM.from_pretrained("monologg/koelectra-base-v3-generator")


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generator_config = config.model_config.generator

        # ELECTRA 의 Discriminator 는 이미 Binary Classifier 임
        # Generator 는 MLM 이지만, 여기서 LM 으로 사용
        #self.critic_model = AutoModelForPreTraining.from_pretrained(self.critic_config.pretrained_model)
        self.model = AutoModelForMaskedLM.from_pretrained(self.generator_config.pretrained_model)
        #self.model = AutoModel.from_pretrained(self.generator_config.pretrained_model)

    def forward(self, inp_ids, attn_mask, token_type_ids):
        return self.model(input_ids=inp_ids,
                    attention_mask=attn_mask,
                    token_type_ids=token_type_ids).logits

    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == '__main__':
    from omegaconf import OmegaConf
    config = OmegaConf.load("/home/taesoo/GEC/config.yaml")
    print(config)
    generator = Generator(config)

    inp_ids = torch.ones((1, 128), dtype=torch.long)
    inp_attn_mask = torch.ones((1, 128), dtype=torch.long)
    inp_token_type = torch.ones((1, 128), dtype=torch.long)

    output = generator.model(input_ids=inp_ids,
                    attention_mask=inp_attn_mask,
                    token_type_ids=inp_token_type)
                    #labels=inp_ids,
                    #output_attentions=inp_attn_mask)
    #loss = generator.model.compute_loss()
    print(output.logits)
# %%
