from transformers import AutoTokenizer, AutoModelForMultipleChoice
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForMultipleChoice.from_pretrained("microsoft/deberta-v3-base", dtype=torch.float32)

