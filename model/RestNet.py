import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat
from timm.layers import create_attn

def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    if issubclass(aa_layer, nn.AvgPool1d):
        return aa_layer(stride)
    else:
        return aa_layer(channels, stride=stride)
    
def get_padding(kernel_size, stride, dilation):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding
    

class Bottleneck1d(nn.Module):
    expansion = 4
    
    def __init__(
        self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64, reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm1d, drop_block=None, drop_path=None, attn_layer=None, aa_layer=None, **kwargs
                 ) -> None:
        super(Bottleneck1d, self).__init__()
        
        width = int(math.floor(planes * (base_width / 64.)) * cardinality)
        first_planes = width // reduce_first 
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)
        
        self.conv1 = nn.Conv1d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        
        self.conv2 = nn.Conv1d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False
        )
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, width, stride=stride, enable=use_aa)
        
        self.conv3 = nn.Conv1d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        
        self.se = create_attn(attn_layer, outplanes)
        
        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path
        
        
    def zero_init_last(self,):
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)
    
    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x
    
def downsample_conv(
    in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None,
):
    norm_layer = norm_layer or nn.BatchNorm1d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)
    
    return nn.Sequential(*[
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
        ])
    
if __name__ == "__main__":
    bn = Bottleneck1d(10, 10)
    x = torch.randn(1, 10, 100)
    print(bn(x).shape)
    
    