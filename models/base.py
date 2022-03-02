#%%
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForPreTraining
#tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-generator")
#model = AutoModelForMaskedLM.from_pretrained("monologg/koelectra-base-v3-generator")


class ELECTRA_GEC_Model(nn.Module):
    def __init__(self, config):
        self.config = config
        self.critic_config = config.model.critic
        self.generator_config = config.model.generator

        # ELECTRA 의 Discriminator 는 이미 Binary Classifier 임
        # Generator 는 MLM 이지만, 여기서 LM 으로 사용
        self.critic_model = AutoModelForPreTraining.from_pretrained(self.critic_config.pretrained_model)
        self.generator_model = AutoModelForMaskedLM.from_pretrained(self.generator_config.pretrained_model)
    
    def inference(self, x):
        # x : [Batch, T]
        loss, logits = self.critic_model(x) # [Batch, T]
        critic_mask = torch.sigmoid(logits) # [Batch, T]

    def foward_critic(self, x):
        return self.critic_model(x)

    def forward_generator(self, x):
        return self.generator_model(x)

    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == '__main__':
    from omegaconf import OmegaConf
    config = OmegaConf.load("/home/taesoo/GEC/config.yaml")
    print(config)
    critic_config = config.model_config.critic
    generator_config = config.model_config.generator

    critic_model = AutoModelForPreTraining.from_pretrained(critic_config.pretrained_model)
    generator_model = AutoModelForMaskedLM.from_pretrained(generator_config.pretrained_model)

    print(generator_config.pretrained_model)
    print(critic_model)
    #print(generator_model) # Word_embedding : 35000, 768

    from torchinfo import summary
    print(summary(critic_model))

    x = torch.ones((1, 128), dtype=torch.int)
    y = critic_model(x)
    print(y)
    print(torch.sigmoid(y.logits))

# %%
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="monologg/koelectra-base-v3-generator",
    tokenizer="monologg/koelectra-base-v3-generator"
)

print(fill_mask("나는 {} 밥을 먹었다.".format(fill_mask.tokenizer.mask_token)))

# %%
