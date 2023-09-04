import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat

class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=19, padding=9)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act_layer = nn.ReLU()
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=4)
        
        
    def forward(self, x):
        return self.avg_pool(self.act_layer(self.bn(self.conv(x))))


class CNN19(nn.Module):
    def __init__(self,):
        super(CNN19, self).__init__()
        # 1 * 30000
        self.down1 = DownLayer(1, 36)
        # 36 * 7500
        self.down2 = DownLayer(36, 36)
        # 36 * 1875
        self.down3 = DownLayer(36, 36)
        # 36 * 469
        self.down4 = DownLayer(36, 36)
        # 36 * 117
        self.down5 = DownLayer(36, 36)
        # 36 * 29
        self.down6 = DownLayer(36, 36)
        # 36 * 7
        self.fc = nn.Linear(36 * 7, 30000)
        
                
    def forward(self, x):
        y = self.down1(x)
        y = self.down2(y)
        y = self.down3(y)
        y = self.down4(y)
        y = self.down5(y)
        y = self.down6(y)
        y = rearrange(y, "b c l -> b (c l)")
        return self.fc(y)
    
class LSTM19(nn.Module):
    def __init__(self, batch_first=False):
        super(LSTM19, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=140, batch_first=batch_first)
        
        
        
    def forward(self, x):
        return
    
    
    
class DRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(DRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        
        
    def forward(self, x):
        return