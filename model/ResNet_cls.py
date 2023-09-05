import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat

from global_utils.torch_utils.layers import Bottleneck1d


class ResNet_cls(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, layers=[2,3,3,3], drop_path=None, zero_init_residual=False):
        super(ResNet_cls, self).__init__()
        self.in_channels = in_channels
        self.channels = [2**(i+5) for i in range(4)]
        self.downsample =   None 
        
        self.init_conv = nn.Conv1d(in_channels, self.channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        
        self.basiclayer1 = nn.Sequential(
            Bottleneck1d(planes=self.channels[0], inplanes=self.channels[0], stride=1, downsample=None, drop_path=drop_path),
            Bottleneck1d(planes=self.channels[0], inplanes=self.channels[0], stride=1, downsample=None, drop_path=drop_path),
        )
        
        self.downsample1 = nn.Conv1d(self.channels[0], self.channels[1], kernel_size=3, stride=2, padding=1, bias=False)
        
        self.Basiclayer2 = nn.Sequential(
            Bottleneck1d(planes=self.channels[1], inplanes=self.channels[1], stride=1, downsample=self.downsample, drop_path=drop_path),
            Bottleneck1d(planes=self.channels[1], inplanes=self.channels[1], stride=1, downsample=None, drop_path=drop_path),
            Bottleneck1d(planes=self.channels[1], inplanes=self.channels[1], stride=1, downsample=None, drop_path=drop_path),
        )
        
        self.downsample2 = nn.Conv1d(self.channels[1], self.channels[2], kernel_size=3, stride=2, padding=1, bias=False)
        
        self.Basiclayer3 = nn.Sequential(
            Bottleneck1d(planes=self.channels[2], inplanes=self.channels[2], stride=1, downsample=self.downsample, drop_path=drop_path),
            Bottleneck1d(planes=self.channels[2], inplanes=self.channels[2], stride=1, downsample=None, drop_path=drop_path),
            Bottleneck1d(planes=self.channels[2], inplanes=self.channels[2], stride=1, downsample=None, drop_path=drop_path),
        )
        
        self.downsample3 = nn.Conv1d(self.channels[2], self.channels[3], kernel_size=3, stride=2, padding=1, bias=False)
        
        self.Basiclayer4 = nn.Sequential(
            Bottleneck1d(planes=self.channels[3], inplanes=self.channels[3], stride=1, downsample=self.downsample, drop_path=drop_path),
            Bottleneck1d(planes=self.channels[3], inplanes=self.channels[3], stride=1, downsample=None, drop_path=drop_path),
            Bottleneck1d(planes=self.channels[3], inplanes=self.channels[3], stride=1, downsample=None, drop_path=drop_path),
        )
        
        
        self.mlp = nn.Sequential(
            nn.Linear(self.channels[3], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.init_conv(x)
        x = self.basiclayer1(x)
        x = self.downsample1(x)
        x = self.Basiclayer2(x)
        x = self.downsample2(x)
        x = self.Basiclayer3(x)
        x = self.downsample3(x)
        x = self.Basiclayer4(x)
        x = torch.mean(x, dim=2)
        x = self.mlp(x)
        return x
    
    
if __name__ == '__main__':
    from torchsummary import summary
    model = ResNet_cls()
    model = model.cuda()
    x = torch.randn(2,2,1000)
    x = x.cuda()
    # y = model(x)
    summary(model, (2,1000))
            