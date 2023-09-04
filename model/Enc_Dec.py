import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os, sys
import wfdb
import time
# sys.path.append('../')
# from  global_utils.utils import random_seed


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


class Seq2Seq(nn.Module):

    def __init__(self) -> None:
        super(Seq2Seq, self).__init__()

        channels = [2**(n + 1) for n in range(5)]
        print(channels)
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

    def forward(self, x):
        encfeature = []
        for i in range(3):
            # print(x.shape)
            x = self.EncList[i](x)
            encfeature.append(x)

        x = self.EncList[3](x)
        # print(x.shape)
        for i in range(3):
            x = self.DecList[i](x)
            # print(x.shape)
            x += encfeature[-(i + 1)]

        return self.DecList[3](x)


noise_path = "../data/mit-bih-noise-stress-test-database-1.0.0/"


def batch_data_iter(num, db, batch_size):
    signal = wfdb.rdrecord(noise_path + '118e24', physical=False).d_signal
    true_sig = wfdb.rdrecord(noise_path + num + 'e00', physical=False).d_signal
    flag = 0
    data_sig = []
    data_true = []
    for i in range(0, 650000, 256):
        if flag < batch_size:
            data_sig.append(torch.FloatTensor(signal[i:i + 256, :]))
            data_true.append(torch.FloatTensor(true_sig[i:i + 256, :]))
            flag += 1
        else:
            yield torch.transpose(torch.stack(data_sig, dim=0), 1,
                                  2).contiguous(), torch.transpose(
                                      torch.stack(data_true, dim=0), 1,
                                      2).contiguous()
            data_sig = []
            data_true = []
            flag = 0


# random_seed(100)

train_num = '118'
train_db = '24'
model = Seq2Seq()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)


def train(epochs, use_gpu=True):
    if use_gpu:
        model = model.cuda()
    ttime = time.time()
    model.train()
    loss_list = []
    loss_total = []
    for inputs, outputs in batch_data_iter(train_num, train_db, 10):
        if use_gpu:
            inputs, outputs = inputs.cuda(), outputs.cuda()
        optimizer.zero_grad()
        pre = model(inputs)
        loss = F.mse_loss(pre, outputs)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    model.eval()
    eval_output = []
    eval_label = []
    for inputs, outputs in batch_data_iter('119', '24', 10):
        if use_gpu:
            inputs, outputs = inputs.cuda(), outputs.cuda()

        pre = model(inputs)
        loss = F.mse_loss(pre, outputs)
        eval_output.append(loss.item())


# ####DEBUG###
a = torch.randn(10, 2, 256)
for i in batch_data_iter('118', '24', 10):
    print(i.shape)
# enb = EncBlock(2, 6, 3)
# print(enb(a).shape)
# nn.Trnas
# db = DecBlock(nn.LazyConvTranspose1d(10, 3, stride=2))
# print(db(a).shape)
# import time
# time.sleep(100)
# import os,sys
# sys.path.append('..')
# from global_utils.mailsend import EmailSender
# se = EmailSender()
# se.get_text("text")
# se.send()

# s2s = Seq2Seq()
# print(s2s(a).shape)
