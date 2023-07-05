import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 

from einops import rearrange, repeat

class Resbottleneck(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_channels, kernel_size,  padding, dilation, act_func=nn.LeakyReLU
        ) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.conv3 = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)        
        self.act_func = act_func()
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.act_func(out)
        out = self.conv2(out)
        out = self.act_func(out)
        out = self.conv3(out)
        out += x
        
        return out
        
class DUblock(nn.Module):
    def __init__(
        self, in_channels, out_channels, du_type
    ):
        super().__init__()
        
        self.du_type = du_type
        
        if du_type == 'D':
            self.du = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        elif du_type == 'U':
            self.du = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            raise ValueError('du_type must be D or U')
    
class Resblock(nn.Module):
    def __init__(
        self,  
        ) -> None:
        super().__init__()
        
