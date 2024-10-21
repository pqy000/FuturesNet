import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args,  data, num_classes=5, num_layers=1):
        super(Model, self).__init__()
        input_dim = data.m
        hidden_dim = args.hidRNN
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # (batch_size, window_size, feature_dim)
        out, _ = self.rnn(x)
        out_last = out[:, -1, :]  # Take the last time step
        out = self.fc(out_last)
        out = torch.softmax(out, dim=1)
        return out