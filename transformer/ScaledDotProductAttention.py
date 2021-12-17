import torch
import torch.nn as nn
import numpy as np

class ScaleDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1,2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


if __name__ == '__main__':
    temp_temperature = 2
    temp = torch.arange(1, 13, dtype=torch.float).view(1, 3, 4)
    temp_q = temp_k = temp_v = temp
    s = ScaleDotProductAttention(temp_temperature)
    print(s.forward(temp_q, temp_k, temp_v))
