import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from siglip_image_tower import SiglipVisionTransformer
from siglip_text_tower import SiglipTextTransformer


class SigLipModel(nn.Module):
    def __init__(self, img_size, dropout_rate):
        super().__init__()
        self.text_model = SiglipTextTransformer()
        self.vision_model = SiglipVisionTransformer(
            img_size=img_size, dropout_rate=dropout_rate
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10.0))
        self.logit_bias = nn.Parameter(torch.ones([]) * -10.0)

    def forward(self, input_ids, attention_mask, pixel_values, return_features=False):
        vision_outputs = self.vision_model(pixel_values)
        text_outputs = self.text_model(input_ids, attention_mask)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        image_features = F.normalize(vision_outputs, dim=-1)
        text_features = F.normalize(text_outputs, dim=-1)
        if return_features:
            return image_features, text_features
        logits = (text_features @ image_features.T) * logit_scale + self.logit_bias
        return logits

    def lock_text_tower(self):
        for param in self.text_model.parameters():
            param.requires_grad = False

    def configure_optimizers(self, weight_decay, learning_rate, ddp=True):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(num_decay_params / num_nodecay_params)

        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=True)
        return optimizer


def loss_fn(logits):
    eye = torch.eye(logits.shape[0], device=logits.device)
    diag = -torch.ones_like(logits) + 2 * eye
    loglike = F.logsigmoid((logits * diag))
    nll = -loglike.sum(dim=-1)
    return nll.mean()


def get_lr(warmup_steps, max_steps, max_lr, min_lr, step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    elif step >= max_steps:
        return min_lr
    else:
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1, "Decay ration should be between 0 and 1"
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


if __name__ == "__main__":
    raise RuntimeError(
        "This file is not meant to be run directly.\n"
        "Import the functions you need instead."
    )
