"""
Character detection and shape detection CNN.
M-Fly 2023-2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class CharacterDetectCNN(nn.Module):

    def __init__(self):
        """Define CNN architecture."""
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.conv4 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.conv5 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.fc_1 = nn.Linear(
            in_features=256, # 2*2*64 in channels
            out_features=128
        )
        self.fc_2 = nn.Linear(
            in_features=128,
            out_features=36
        )
        self.relu = nn.ReLU()

        self.init_weights()

        # self.conv1 = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=4,
        #     kernel_size=5,
        #     stride=1,
        #     padding=2
        # )
        # self.pool = nn.MaxPool2d(
        #     kernel_size=2,
        #     stride=2
        # )
        # self.conv2 = nn.Conv2d(
        #     in_channels=4,
        #     out_channels=8,
        #     kernel_size=5,
        #     stride=1,
        #     padding=2
        # )
        # self.conv3 = nn.Conv2d(
        #     in_channels=8,
        #     out_channels=16,
        #     kernel_size=5,
        #     stride=1,
        #     padding=2
        # )
        # self.conv4 = nn.Conv2d(
        #     in_channels=16,
        #     out_channels=32,
        #     kernel_size=5,
        #     stride=1,
        #     padding=2
        # )
        # self.conv5 = nn.Conv2d(
        #     in_channels=32,
        #     out_channels=64,
        #     kernel_size=5,
        #     stride=1,
        #     padding=2
        # )
        # self.fc_1 = nn.Linear(
        #     in_features=1024, # 8x8x32 in channels
        #     out_features=256
        # )
        # self.fc_2 = nn.Linear(
        #     in_features=256,
        #     out_features=36
        # )
        # self.relu = nn.ReLU()

        # self.init_weights()

    def init_weights(self):
        """Initialize weights and biases for each layer."""
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        for fc in [self.fc_1, self.fc_2]:
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(C_in))
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):
        """Define forward propogation through each layer."""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x


class CharacterDetectCNNSmall(nn.Module):

    def __init__(self):
        """Define CNN architecture."""
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.conv4 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.conv5 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
        )
        # self.fc_1 = nn.Linear(
        #     in_features=4096, # 128x128 input size
        #     out_features=256
        # )
        self.fc_1 = nn.Linear(
            in_features=1024, # 8x8x32 in channels
            out_features=256
        )
        self.fc_2 = nn.Linear(
            in_features=256,
            out_features=7
        )
        self.relu = nn.ReLU()
        # self.cnn_dropout = nn.Dropout(0.1)
        # self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        """Initialize weights and biases for each layer."""
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        for fc in [self.fc_1, self.fc_2]:
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(C_in))
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):
        """Define forward propogation through each layer."""
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.cnn_dropout(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        # x = self.cnn_dropout(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        # x = self.cnn_dropout(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.relu(x)
        # x = self.cnn_dropout(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.relu(x)
        # x = self.cnn_dropout(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        # x = self.dropout(x)
        x = self.fc_2(x)

        return x


class ShapeDetectCNN(nn.Module):

    def __init__(self):
        """Define CNN architecture."""
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.fc_1 = nn.Linear(
            in_features=4096, # 128x128 input size
            out_features=256
        )
        self.fc_2 = nn.Linear(
            in_features=256,
            out_features=7
        )
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        """Initialize weights and biases for each layer."""
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        for fc in [self.fc_1, self.fc_2]:
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(C_in))
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):
        """Define forward propogation through each layer."""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x
