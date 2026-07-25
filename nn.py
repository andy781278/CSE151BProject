import torch
import torch.nn as nn

class ffn(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, 100)
        self.l2 = nn.Linear(100,100)
        self.l3 = nn.Linear(100, 1)

    def forward(self, x, mask):
        samples = x.shape[0]
        labels = x.shape[1]
        vocab = x.shape[2]

        flat = x.view(samples * labels, vocab)

        x1 = self.relu(self.l1(flat))
        x2 = self.relu(self.l2(x1))
        score = self.l3(x2)

        score = score.view(samples, labels)
        return score.masked_fill(mask == 0, float("-inf"))