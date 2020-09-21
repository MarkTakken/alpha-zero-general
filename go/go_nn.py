from typing import List, Union, Tuple

import numpy as np

import torch
import math
from torch import Tensor, nn

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int=17,
                 out_channels: int=256):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1), #changed outchannels = 256 to outchannels = outchannels
            nn.BatchNorm2d(num_features=out_channels),  #Mark 5/22: Changed 256 to out_channels
            nn.ReLU(inplace = True),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.extractor(x)

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int=256, #Mark 5/21: Should default to 256 (instead of 17) because it receives as input the output of the convolutional block?
                 inter_channels: int=256,
                 out_channels: int=256):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inter_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: Tensor) -> Tensor:
        #Not in othello implementation, so commented out for now
        #if torch.cuda.is_available():
        #    self.cuda()
        x_features = self.extractor(x)
        x_out = x + x_features
        return self.relu(x_out) #Mark 5/21: Changed x to x_out

class Flatten(nn.Module):
    def __init__(self,
                 row_length):
        super().__init__()
        self.row_length = row_length
    def forward(self,x):
        return x.view(-1,self.row_length)

class ToScalar(nn.Module):
    def forward(self,x):
        return x.view(-1)

class PolicyHead(nn.Module):
    def __init__(self,
                 in_channels : int = 256,
                 board_size: int = 19,
                 conv_out_channels: int = 2):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = conv_out_channels, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(num_features = conv_out_channels),
            nn.ReLU(inplace = True),
            Flatten(row_length = board_size*board_size*conv_out_channels),
            nn.Linear(board_size**2*conv_out_channels,board_size**2+1),
            nn.Softmax(dim = 1)
            #Add softmax?
        )

    def forward(self,x):
        return self.extractor(x)

class ValueHead(nn.Module):
    def __init__(self,
                 in_channels : int = 256,
                 board_size: int = 19,
                 conv_out_channels: int = 1,
                 hidden_layer_size: int = 256):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = conv_out_channels, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(num_features = conv_out_channels),
            nn.ReLU(inplace = True),
            Flatten(row_length=board_size*board_size*conv_out_channels),
            nn.Linear(board_size**2*conv_out_channels,hidden_layer_size),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_layer_size,1),
            nn.Tanh(),
            ToScalar()
        )

    def forward(self,x):
        return self.extractor(x)

class GoNNet(nn.Module):
    def __init__(self,
                 board_size: int=19, #Mark 5/21: Changed 9 to 19
                 history: int=8,
                 n_blocks: int=3,  # Alphago Zero uses 19 or 39 blocks, we use fewer. 
                 n_filters: int=256):
        super().__init__()
        self.in_channels = history * 2 + 1
        self.conv_block = ConvBlock(in_channels=self.in_channels, out_channels=n_filters)
        blocks: List[ResidualBlock] = []
        for _ in range(n_blocks):
            resblock = ResidualBlock(in_channels=n_filters, out_channels=n_filters)
            blocks.append(resblock)
        self.blocks = nn.Sequential(*blocks)
        self.policy_head = PolicyHead(in_channels = n_filters, board_size = board_size)
        self.value_head = ValueHead(in_channels = n_filters, board_size = board_size)

    def forward(self, x: Union[np.ndarray, Tensor]) -> Tuple[Tensor, Tensor]:
        # Convert to tensor if necessary
        x = torch.as_tensor(x)
        x.requires_grad_(True)
        test_if_nan(x[0][0][0][0],0)
        x = self.conv_block(x)
        test_if_nan(x[0][0][0][0],1)
        x = self.blocks(x)
        test_if_nan(x[0][0][0][0],2)
        pi = self.policy_head(x)
        test_if_nan(pi[0][0],3)
        v = self.value_head(x)
        test_if_nan(v[0],4)
        return (pi, v)

def test_if_nan(x,n):
    if math.isnan(x):
        print(n)

def test():
    network = GoNNet(board_size=4,history=1)
    canonical_state = np.array([[[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]],[[0,1,0,0],[1,1,0,0],[0,0,0,1],[1,0,0,0]],[[0,0,0,0],[0,0,1,1],[1,0,0,0],[0,1,1,0]]]])
    print(canonical_state)
    print('------------')
    print(network(canonical_state))
