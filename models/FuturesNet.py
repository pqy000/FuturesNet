# # from tsai.models import InceptionTimePlus
# from tsai.models.InceptionTimePlus import InceptionTimePlus
# from torch import nn
# import torch
#
# class Model(nn.Module):
#     def __init__(self, args, data, num_classes=5, pretrained=True):
#         super(Model, self).__init__()
#
#         # 使用 InceptionTimePlus 预训练模型
#         input_dim = data.m
#         self.feature_extractor = InceptionTimePlus(c_in=input_dim, c_out=num_classes)
#         self.fc = nn.Linear(num_classes, num_classes)  # 这里假设 num_classes 一致
#
#     def forward(self, x):
#         # x的输入维度为 (batch_size, window_size, feature_dim)
#         x = x.transpose(1, 2)
#         out = self.feature_extractor(x)
#         out = self.fc(out)
#         out = torch.softmax(out, dim=1)
#         return out
#

from tsai.models.InceptionTimePlus import InceptionTimePlus
from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, args, data, num_classes=5, hidden_size=64, num_layers=1, pretrained=True):
        super(Model, self).__init__()

        # 使用 InceptionTimePlus 预训练模型
        input_dim = data.m
        self.feature_extractor = InceptionTimePlus(c_in=input_dim, c_out=args.window*num_classes)

        # Adding LSTM after InceptionTimePlus
        self.lstm = nn.LSTM(input_size=num_classes, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=False)

        # Fully connected layer for classification, assuming hidden_size from LSTM
        self.fc = nn.Linear(hidden_size, num_classes)  # Add another layer for classification, assuming
        # Optional skip connection from InceptionTimePlus to final classification
        self.skip_fc = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        # x 的输入维度为 (batch_size, window_size, feature_dim)
        batch_size, window_size, feature_dim = x.shape[0], x.shape[1], x.shape[2]
        x = x.transpose(1, 2)  # Transpose to match the expected input for InceptionTimePlus

        # InceptionTimePlus feature extraction
        inception_out = self.feature_extractor(x)  # Shape: (batch_size, window_size, num_classes)
        inception_out = inception_out.view(batch_size, window_size, -1)  # Flatten to (batch_size, window)
        lstm_out, (hidden_state, _) = self.lstm(inception_out)  # hidden_state shape: (num_layers, batch_size, hidden_size)

        # Take the last hidden state
        final_hidden_state = hidden_state[-1]  # Shape: (batch_size, hidden_size)

        # Optional skip connection (combining features from InceptionTimePlus output)
        skip_out = self.skip_fc(inception_out[:, -1, :])  # Take last output from InceptionTimePlus

        # Final output (LSTM + Skip connection)
        out = self.fc(final_hidden_state) + skip_out  # Combine LSTM final output with skip connection

        # Softmax for classification
        out = torch.softmax(out, dim=1)
        return out