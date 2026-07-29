import torch
import torch.nn as nn
from transformers import AutoModelForMultipleChoice

class ARCModel(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-large", grad_ckpt=False):
        super().__init__()
        self.encoder = AutoModelForMultipleChoice.from_pretrained(model_name)
        if grad_ckpt:
            self.encoder.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, choice_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits

        if choice_mask is not None:
            logits = logits.masked_fill(
                ~choice_mask, torch.finfo(logits.dtype).min
            )
        return logits