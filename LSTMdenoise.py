import sys

sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat
from global_utils.submodels import SE_block1d


class DRblock(nn.Module):

    def __init__(self,
                 channels,
                 hidden_size,
                 SE_hidden_size,
                 conv_kernel_size,
                 act_layer=nn.GELU,
                 se_act_layer=nn.GELU):
        super(DRblock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=channels,
                               out_channels=hidden_size,
                               kernel_size=conv_kernel_size,
                               padding=(conv_kernel_size - 1) // 2)
        self.act_layer = act_layer()
        self.conv2 = nn.Conv1d(hidden_size,
                               channels,
                               conv_kernel_size,
                               padding=(conv_kernel_size - 1) // 2)
        self.seblock = SE_block1d(channels=channels,
                                  hidden_size=SE_hidden_size,
                                  act_layer=se_act_layer)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act_layer(y)
        y = self.conv2(y)
        y = self.seblock(y)
        y = x + y
        return self.act_layer(y)


class DRLayer(nn.Module):

    def __init__(self,
                 kernel_sizes,
                 in_channels,
                 out_channels,
                 channels,
                 hidden_size,
                 SE_hidden_size,
                 act_layer=nn.GELU,
                 se_act_layer=nn.GELU):
        super(DRLayer, self).__init__()
        self.modulelist = nn.ModuleList()
        for conv_size in kernel_sizes:
            module_seq = nn.Sequential(
                nn.Conv1d(in_channels,
                          channels,
                          conv_size,
                          padding=(conv_size - 1) // 2),
                DRblock(channels, hidden_size, SE_hidden_size, conv_size,
                        act_layer, se_act_layer),
                DRblock(channels, hidden_size, SE_hidden_size, conv_size,
                        act_layer, se_act_layer),
                DRblock(channels, hidden_size, SE_hidden_size, conv_size,
                        act_layer, se_act_layer),
                nn.Conv1d(channels,
                          out_channels,
                          conv_size,
                          padding=(conv_size - 1) // 2),
            )
            self.modulelist.add_module("net_{}".format(conv_size), module_seq)

    def forward(self, x):
        y_list = []
        for seq in self.modulelist:
            y_list.append(seq(x))
        y = torch.stack(y_list, dim=0)
        return torch.sum(y, dim=0)


class Context_Contrast(nn.Module):
    def __init__(self, in_channels=48, out_channels=96, kernel_size=5, dilation=10):
        super(Context_Contrast, self).__init__()
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size,
                      padding=dilation * (kernel_size - 1) // 2,
                      dilation=10),
            nn.GELU(),
            nn.Conv1d(out_channels,
                      out_channels,
                      kernel_size,
                      padding=dilation * (kernel_size - 1) // 2,
                      dilation=10),
            nn.GELU(),
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size,
                      padding=(kernel_size - 1) // 2),
            nn.GELU(),
            nn.Conv1d(out_channels,
                      out_channels,
                      kernel_size,
                      padding=(kernel_size - 1) // 2),
            nn.GELU(),
        )

    def forward(self, x):
        y1 = self.dilated_conv(x)
        y2 = self.local_conv(x)
        y3 = y2 - y1
        return torch.cat([y1, y2, y3], dim=1)


class U_net(nn.Module):
    def __init__(self, act_layer=nn.GELU):
        super(U_net, self).__init__()
        # enc1 
        self.con11 = nn.Conv1d(1, 12, kernel_size=20)
        self.con12 = nn.Conv1d(12, 12, kernel_size=20)
        self.con13 = nn.Conv1d(12, 12, kernel_size=20)
        self.maxpool1 = nn.MaxPool1d(kernel_size=10, stride=10)
        
        # enc2 
        self.con21 = nn.Conv1d(12, 32, kernel_size=10)
        self.con22 = nn.Conv1d(32, 32, kernel_size=10)
        self.con23 = nn.Conv1d(32, 32, kernel_size=10)
        self.maxpool2 = nn.MaxPool1d(5, stride=5)
        
        # enc3
        self.con31 = nn.Conv1d(32, 48, kernel_size=5)
        self.con32 = nn.Conv1d(48, 48, kernel_size=5)
        self.con33 = nn.Conv1d(48, 48, kernel_size=5)
        self.maxpool2 = nn.MaxPool1d(2, stride=2)
        
        # enc4 
        self.contrast_context = Context_Contrast()
        
        
        # dec4 
        self.upsample4 = nn.Upsample(scale_factor=2)
        
        # dec3 
        self.transconv33 = nn.ConvTranspose1d(48 + 288, 48, 5)
        self.transconv32 = nn.ConvTranspose1d(48, 48, 5)
        self.transconv31 = nn.ConvTranspose1d(48, 48, 5)
        self.upsample3 = nn.Upsample(scale_factor=5)
        
        # dec2
        self.transconv23 = nn.ConvTranspose1d(32 + 48, 32, 10)
        self.transconv22 = nn.ConvTranspose1d(32, 32, 10)
        self.transconv21 = nn.ConvTranspose1d(32, 32, 10)
        self.upsample2 = nn.Upsample(scale_factor=10)
        
        # dec1
        self.transconv13 = nn.ConvTranspose1d(12 + 32, 12, 20)
        self.transconv12 = nn.ConvTranspose1d(12, 12, 20)
        self.transconv11 = nn.ConvTranspose1d(12, 12, 20) 
        
        self.conv1d = nn.Conv1d(12, 1, 1)
               
        self.act_layer = act_layer()
        
        
                

    def forward(self, x):
        
        
        
        return


# #
# drblock = DRblock(10, 20, 20, 3)
# a = torch.randn(32,10,250)
# print(drblock(a).shape)
# drlayer = DRLayer([3,5,9], 1, 1, 32, 4*32, 4*32)
a = torch.randn(32, 48, 256)
cc = Context_Contrast()
print(cc(a).shape)
# print(drlayer(a).shape)