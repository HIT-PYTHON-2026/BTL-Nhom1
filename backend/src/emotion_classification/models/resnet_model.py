import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.downsample(shortcut)
        x = self.relu(x)

        return x


class ResNet18(nn.Module):
    def __init__(self, residual_block, n_blocks_lst, n_classes):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self.create_layer(
            residual_block,
            64, 64,
            n_blocks_lst[0], 1
        )
        self.layer2 = self.create_layer(
            residual_block,
            64, 128,
            n_blocks_lst[1], 2
        )
        self.layer3 = self.create_layer(
            residual_block,
            128, 256,
            n_blocks_lst[2], 2
        )
        self.layer4 = self.create_layer(
            residual_block,
            256, 512,
            n_blocks_lst[3], 2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.flatten = nn.Flatten()
        # self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )

    def create_layer(self, residual_block, in_channels, out_channels, n_blocks, stride):
        blocks = []
        first_block = residual_block(in_channels, out_channels, stride)
        blocks.append(first_block)

        for idx in range(1, n_blocks):
            block = residual_block(out_channels, out_channels, stride)
            blocks.append(block)

        block_sequential = nn.Sequential(*blocks)
        return block_sequential

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.maxpool(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.flatten(x)
        # x = self.dropout(x)
        x = self.fc(x)

        return x
