import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F


# 自定义模型：卷积特征提取 + LSTM + Skip Connection
class Model(nn.Module):
    def __init__(self, args, data, num_classes=5, pretrained=True):
        super(Model, self).__init__()
        # 使用预训练的卷积模型 (例如ResNet) 进行特征提取
        feature_dim = data.m
        hidden_dim = args.hidRNN
        resnet = models.resnet18(pretrained=pretrained)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # 去掉全连接层

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        # Skip Connection
        self.skip_connection = nn.Linear(32, hidden_dim)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, window_size, feature_dim = x.size()
        # 扩展维度，适配ResNet的输入 (batch_size, 1, window_size, feature_dim)
        # x = x.view(batch_size * window_size, feature_dim, 1, 1)  # 为卷积输入做调整
        x = x.unsqueeze(1)  # 添加通道维度

        x = self.feature_extractor(x)  # 卷积特征提取
        x = x.view(batch_size, window_size, -1)  # 调整维度为 (batch, window, 512)

        skip_out = self.skip_connection(x)  # Skip Connection on hidden state

        # LSTM for temporal modeling with skip connection
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out + skip_out
        out = self.fc(lstm_out[:, -1, :])  # 取最后一个时刻的输出
        out = torch.softmax(out, dim=1)
        return out