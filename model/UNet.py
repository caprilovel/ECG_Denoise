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
        # du_type: D for downsampling, U for upsampling
        self.du_type = du_type
        
        if du_type == 'D':
            self.du = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        elif du_type == 'U':
            self.du = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            raise ValueError('du_type must be D or U')
    

class EncBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=1,
                 stride=2,
                 use_relu=True):
        super(EncBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride)
        self.bn = nn.LazyBatchNorm1d()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DecBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=2,
                 padding=1,
                 use_relu=True):
        super(DecBlock, self).__init__()
        self.use_relu = use_relu

        self.conv = nn.ConvTranspose1d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding)
        self.bn = nn.LazyBatchNorm1d()
        if use_relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        if self.use_relu:
            return self.relu(self.bn(self.conv(x)))
        else:
            return self.bn(self.conv(x))


class UNet(nn.Module):
    def __init__(self) -> None:
        super(UNet, self).__init__()

        channels = [2**(n + 1) for n in range(5)]
        self.EncList = nn.ModuleList()
        self.DecList = nn.ModuleList()

        for i in range(4):
            self.EncList.append(EncBlock(channels[i], channels[i + 1], 3))
            if i != 3:
                self.DecList.append(
                    DecBlock(channels[-(i + 1)], channels[-(i + 2)], 4))
            else:
                self.DecList.append(
                    DecBlock(channels[-(i + 1)],
                             channels[-(i + 2)],
                             4,
                             use_relu=False))
                
        self.bottleneck = nn.Sequential(
            nn.Conv1d(channels[4], channels[4], 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(channels[4]),
            nn.Conv1d(channels[4], channels[4], 3 , padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(channels[4]),
            nn.Conv1d(channels[4], channels[4], 1),
        )

    def forward(self, x):
        encfeature = []
        for i in range(3):
            # print(x.shape)
            x = self.EncList[i](x)
            encfeature.append(x)

        x = self.EncList[3](x)
        x = self.bottleneck(x) + x 
        # print(x.shape)
        for i in range(3):
            x = self.DecList[i](x)
            # print(x.shape)
            x += encfeature[-(i + 1)]

        return self.DecList[3](x)


        
if __name__ == "__main__":
    model = UNet()
    x = torch.randn(32, 2, 256)
    y = model(x)
    print(y.shape)