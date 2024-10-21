import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data, num_classes=5):
        super(Model, self).__init__()
        # Conv1D layers
        input_dim = data.m
        window_size = args.window
        cnn_output_dim = args.hidCNN
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_output_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=cnn_output_dim, out_channels=cnn_output_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(cnn_output_dim * (window_size // 2), num_classes)

    def forward(self, x):
        # (batch_size, window_size, feature_dim) -> (batch_size, feature_dim, window_size)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        x = torch.softmax(x, dim=1)
        return x