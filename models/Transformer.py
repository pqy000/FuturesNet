import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data, num_classes=5, num_heads=5, hidden_dim=64):
        super(Model, self).__init__()
        # Self-Attention Encoder
        input_dim = data.m
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # (batch_size, window_size, feature_dim)
        attn_output, _ = self.attn(x, x, x)
        out_last = attn_output[:, -1, :]  # Take the last time step
        out = self.fc(out_last)
        out = torch.softmax(out, dim=1)
        return out