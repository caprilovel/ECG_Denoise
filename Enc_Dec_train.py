import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import wfdb
import os
import matplotlib.pyplot as plt
import random
import math

path = "../data/mit-bih-noise-stress-test-database-1.0.0/"


def ecg_noise_add(data, alpha=0.1, random_seed=500):
    random.seed(random_seed)
    datashape = data.shape
    noise_index = ['bw', 'ma', 'em']
    length = data.shape[-2]
    if len(data.shape) is 2:
        data = np.expand_dims(data, 0)
    noise_list = []
    for i in range(data.shape[0]):
        group_index = random.randint(0, len(noise_index) - 1)
        data_index = random.randint(0, 650000 - int(data.shape[-2]) - 1)
        noise_list.append(
            alpha *
            wfdb.rdrecord(path + noise_index[group_index],
                          physical=False).d_signal[data_index:data_index +
                                                   int(data.shape[-2])])
    output = data + np.stack(noise_list, 0)
    if output.shape != datashape:
        output = np.squeeze(output, 0)
    return output


data_path = "../data/mit-bih-arrhythmia-database-1.0.0/"

noise_path = "../data/mit-bih-noise-stress-test-database-1.0.0/"
arryth_path = "../data/mit-bih-arrhythmia-database-1.0.0/"


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


def batch_arrythdata_iter(nums, batch_size):
    for num in nums:
        signal = wfdb.rdrecord(arryth_path + str(num), physical=False).d_signal
        flag = 0
        data_sig = []
        for i in range(0, 650000, 256):
            if flag < batch_size:
                data_sig.append(signal[i:i + 256, :])
                flag += 1
            else:
                ground_truth_data = np.stack(data_sig, 0)
                yield torch.transpose(
                    torch.FloatTensor(ecg_noise_add(ground_truth_data)), 1,
                    2).contiguous(), torch.transpose(
                        torch.FloatTensor(ground_truth_data), 1,
                        2).contiguous()
                data_sig = []
                flag = 0


class APReLU(nn.Module):

    def __init__(self, channels):
        super(APReLU, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        fcnList = [
            nn.Linear(2 * channels, 2 * channels),
            nn.BatchNorm1d(2 * channels),
            nn.ReLU(),
            nn.Linear(2 * channels, channels),
            nn.BatchNorm1d(channels),
            nn.Sigmoid()
        ]
        self.fcn = nn.Sequential(*fcnList)

    def forward(self, x):
        zerox = torch.zeros_like(x)
        posx = torch.max(x, zerox)
        negx = torch.min(x, zerox)

        concatx = torch.concat(
            [self.gap(posx).squeeze(-1),
             self.gap(negx).squeeze(-1)], dim=1)
        concatx = self.fcn(concatx)
        return posx + concatx.unsqueeze(2) * negx


class EncoderCell(nn.Module):

    def __init__(
        self,
        out_channels,
        kernel_size,
        padding,
        stride=2,
    ):
        super(EncoderCell, self).__init__()
        self.conv = nn.LazyConv1d(out_channels,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  stride=stride)
        self.aprelu = APReLU(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.aprelu(out)
        out = self.bn(out)
        # print(out.shape)
        return out


class DeNoiseEnc(nn.Module):

    def __init__(self, ):
        super(DeNoiseEnc, self).__init__()
        self.conv_kernel = [17, 17, 3, 3]
        self.paddingsize = [8, 8, 1, 1]
        self.out_channels = [4, 8, 16, 32]
        self.EncoderList = nn.ModuleList()
        for i in range(4):
            self.EncoderList.add_module(
                "cell{}".format(i),
                EncoderCell(self.out_channels[i], self.conv_kernel[i],
                            self.paddingsize[i]))

    def forward(self, x):
        out = []
        for cell in self.EncoderList:
            x = cell(x)
            out.append(x)
        return out


class DAM(nn.Module):

    def __init__(self, channels):
        super(DAM, self).__init__()
        # Channel Attention
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        fcnList = [
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.Sigmoid()
        ]
        self.fcn1 = nn.Sequential(*fcnList)
        self.fcn2 = nn.Sequential(*fcnList)

        # Spatial Attention
        self.cap = nn.AdaptiveAvgPool1d(1)
        self.cmp = nn.AdaptiveMaxPool1d(1)
        self.convsa = nn.Conv1d(2, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_transpose = x.transpose(1, 2)
        # Channel Attention
        gapx = self.gap(x).squeeze(2)
        gmpx = self.gmp(x).squeeze(2)
        gapx = self.fcn1(gapx)
        gmpx = self.fcn2(gmpx)
        Cattn = self.sigmoid(gapx + gmpx).unsqueeze(-1)

        #Spatial Attn
        capx = self.cap(x_transpose).transpose(1, 2)
        cmpx = self.cmp(x_transpose).transpose(1, 2)
        catcp = torch.cat((capx, cmpx), dim=1)
        Sattn = self.sigmoid(self.convsa(catcp).squeeze(1)).unsqueeze(-2)
        x = Cattn * x
        x = Sattn * x
        return x


class DecoderCell(nn.Module):

    def __init__(self,
                 out_channels,
                 kernel_size,
                 padding,
                 stride=2,
                 last=False):
        super(DecoderCell, self).__init__()
        self.last = last

        self.deconv = nn.LazyConvTranspose1d(out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             padding=padding,
                                             stride=stride)
        self.aprelu = APReLU(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

        if last == False:
            self.dam = DAM(out_channels)

    def forward(self, x):
        outx = self.deconv(x)
        outx = self.aprelu(outx)
        outx = self.bn(outx)
        if self.last is not True:
            outx = self.dam(outx)
        # print(outx.shape)
        return outx


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

        self.bn_output = nn.LazyBatchNorm1d()

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


class DeNoiseDec(nn.Module):

    def __init__(self, ):
        super(DeNoiseDec, self).__init__()
        self.conv_kernel = [4, 4, 18, 18]
        self.paddingsize = [1, 1, 8, 8]
        self.out_channels = [16, 8, 4, 2]
        DecoderList = []
        for i in range(4):
            if i is not 3:
                DecoderList.append(
                    DecoderCell(self.out_channels[i], self.conv_kernel[i],
                                self.paddingsize[i]))
            else:
                DecoderList.append(
                    DecoderCell(self.out_channels[i],
                                self.conv_kernel[i],
                                self.paddingsize[i],
                                last=True))
        self.DecoderList = nn.ModuleList(DecoderList)

    def forward(self, xlist):
        y3 = self.DecoderList[0](xlist[-1])
        y2 = self.DecoderList[1](y3 + xlist[-2])
        y1 = self.DecoderList[2](y2 + xlist[-3])
        y0 = self.DecoderList[3](y1 + xlist[-4])

        return y0


class Seq2Seq2(nn.Module):

    def __init__(self, ):
        super(Seq2Seq2, self).__init__()
        self.enc = DeNoiseEnc()
        self.dec = DeNoiseDec()

    def forward(self, x):
        return self.dec(self.enc(x))


# import torch.optim as optim

# train_num = '118'
# train_db = '24'
# dataset_divide = [0.6, 0.2, 0.2]
# DS1 = [
#     101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205,
#     207, 208, 209, 215, 220, 223, 230
# ]
# DS2 = [
#     100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219,
#     221, 222, 228, 231, 232, 233, 234
# ]
# Total_DS = DS1 + DS2
# random.shuffle(Total_DS)
# DS_train = Total_DS[0:int(dataset_divide[0] * len(Total_DS))]
# DS_eval = Total_DS[int(dataset_divide[0] *
#                        len(Total_DS)):int((dataset_divide[1] +
#                                            dataset_divide[0]) * len(Total_DS))]
# DS_test = Total_DS[int((dataset_divide[1] + dataset_divide[0]) *
#                        len(Total_DS)):]
# save_path = "./model_data_pth.tar"
# import time

# model = Seq2Seq()

# optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)


# def train(epochs, model, batch_size=20, use_gpu=True):
#     print("init train!")
#     if use_gpu:
#         model = model.cuda()
#     ttime = time.time()
#     model.train()
#     loss_list = []
#     eval_list = []
#     for epoch in range(epochs):
#         loss_total = []
#         for inputs, outputs in batch_arrythdata_iter(DS_train, batch_size):
#             if use_gpu:
#                 inputs, outputs = inputs.cuda(), outputs.cuda()
#             # print(inputs.shape)
#             optimizer.zero_grad()
#             pre = model(inputs)
#             # print(pre.shape)
#             loss = F.mse_loss(pre, outputs)
#             loss_total.append(loss.item())
#             loss.backward()
#             optimizer.step()
#         loss_list.append(sum(loss_total))
#         if epoch % 5 == 0:
#             torch.save(model, "./model_enc_dec_data_epoch{}.pth".format(epoch))
#         model.eval()

#         eval_mse = []
#         for inputs, outputs in batch_arrythdata_iter(DS_eval, batch_size):
#             if use_gpu:
#                 inputs, outputs = inputs.cuda(), outputs.cuda()

#             pre = model(inputs)
#             loss = F.mse_loss(pre, outputs)
#             eval_mse.append(loss.item())
#         eval_mse_value = sum(eval_mse)
#         print(eval_mse_value)
#         eval_list.append(eval_mse_value)
#     return eval_list


# list_mse = train(100, model)
# plt.plot(list_mse)
# plt.save('loss.png')