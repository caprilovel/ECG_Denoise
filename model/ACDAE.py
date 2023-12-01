import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat


class ECA_module(nn.Module):
    def __init__(self, Channels, k_size = 3):
        super(ECA_module, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2))
        y = y.transpose(-1, -2)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class EncBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super(EncBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.pool(self.conv(x)))


class DecBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super(DecBlock, self).__init__()

        self.conv = nn.ConvTranspose1d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size = kernel_size,
                                       padding=(kernel_size - 1) // 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')
        self.relu = nn.LeakyReLU()

        self.ECA = ECA_module(Channels=out_channels)
    def forward(self, x):
        return self.ECA(self.relu(self.upsample(self.conv(x))))


class ACDAE(nn.Module):
    def __init__(self) -> None:
        super(ACDAE, self).__init__()

        channels = [2, 16, 32, 64, 128]
        Kernal_Size = [13, 7, 7, 7]
        self.EncList = nn.ModuleList()
        self.DecList = nn.ModuleList()

        for i in range(4):
            self.EncList.append(EncBlock(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=Kernal_Size[i]))
            self.DecList.append(DecBlock(in_channels=channels[-(i + 1)], out_channels=channels[-(i + 2)], kernel_size=Kernal_Size[-(i+1)]))

    def forward(self, x):
        encfeature = []
        for i in range(3):
            x = self.EncList[i](x)
            encfeature.append(x)

        x = self.EncList[3](x)

        for i in range(3):
            x = self.DecList[i](x)
            x += encfeature[-(i + 1)]
        return self.DecList[3](x)
