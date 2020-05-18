from typing import List

import numpy as np

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int=17,
                 out_channels: int=256):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: Tensor) -> Tensor:
        x_features = self.extractor(x)
        x_out = x + x_features
        return self.relu(x)


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int=17,
                 out_channels: int=256):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.extractor(x)


class GoCNN(nn.Module):
    def __init__(self,
                 board_size: int=9,
                 history_size_per_player: int=8,
                 n_blocks: int=3,  # alphago Zero uses 19 or 39 blocks, we use less. 
                 n_filters: int=256):
        self.board_size = board_size
        self.history_size_per_player = history_size_per_player
        self.in_channels = history_size_per_player * 2 + 1
        self.input_shape = [board_size, board_size, self.in_channels]
        
        self.conv_block = ConvBlock(in_channels=self.in_channels, out_channels=256)
        
        self.blocks: List[ResidualBlock] = []
        for i in range(n_blocks):
            resblock = ResidualBlock(in_channels=n_filters, out_channels=n_filters)
            self.blocks.append(resblock)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_block(x)
        for block in self.blocks:
            x = block(x)
        return x


class PolicyHead(nn.Module):
    raise NotImplementedError("TODO:")

class ValueHead(nn.Module):
    raise NotImplementedError("TODO:")
