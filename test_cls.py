import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat

from torch.utils.data import Dataset, DataLoader

from model.ResNet_cls import ResNet_cls


def acc(pred, label):
    pred = torch.argmax(pred, dim=1)
    return torch.sum(pred == label).item() / len(label)


def precision(pred, label):
    pred = torch.argmax(pred, dim=1)
    TP = torch.sum(pred * label).item()
    FP = torch.sum(pred * (1 - label)).item()
    return TP / (TP + FP)

def f1_score(pred, label):
    pred = torch.argmax(pred, dim=1)
    TP = torch.sum(pred * label).item()
    FP = torch.sum(pred * (1 - label)).item()
    FN = torch.sum((1 - pred) * label).item()
    return TP / (TP + 0.5 * (FP + FN))

class ClsDataset(Dataset):
    def __init__(self, train=True, noised=False) -> None:
        super().__init__()
        if not noised:
            if train:
                self.N_data = np.load('./data/cls_data/N_train_data.npy')
                self.V_data = np.load('./data/cls_data/V_train_data.npy')
                self.data = np.concatenate((self.N_data, self.V_data), axis=0)
                self.label = np.concatenate((np.zeros(len(self.N_data)), np.ones(len(self.V_data))), axis=0)
            else:
                self.N_data = np.load('./data/cls_data/N_test_data.npy')
                self.V_data = np.load('./data/cls_data/V_test_data.npy')
                self.data = np.concatenate((self.N_data, self.V_data), axis=0)
                self.label = np.concatenate((np.zeros(len(self.N_data)), np.ones(len(self.V_data))), axis=0)
        else:
            if train:
                self.N_data = np.load('./data/cls_data/noised_N_train_data.npy')
                self.V_data = np.load('./data/cls_data/noised_V_train_data.npy')
                self.data = np.concatenate((self.N_data, self.V_data), axis=0)
                self.label = np.concatenate((np.zeros(len(self.N_data)), np.ones(len(self.V_data))), axis=0)
            else:
                self.N_data = np.load('./data/cls_data/noised_N_test_data.npy')
                self.V_data = np.load('./data/cls_data/noised_V_test_data.npy')
                self.data = np.concatenate((self.N_data, self.V_data), axis=0)
                self.label = np.concatenate((np.zeros(len(self.N_data)), np.ones(len(self.V_data))), axis=0)
        
            
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)
    

test_dataset = ClsDataset(train=False)
noised_test_dataset = ClsDataset(train=False, noised=True)

test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
noised_test_dataloader = DataLoader(noised_test_dataset, batch_size=128, shuffle=False)

model = ResNet_cls()
model.load_state_dict(torch.load('./model/cls_model.pth'))
model = model.cuda()

from model.UNet import UNet
denoised_model1 = UNet()
denoised_model1.load_state_dict(torch.load('./model_save/UNet/UNet_99_emb_intensity-4.pth'))
denoised_model1 = denoised_model1.cuda()

from model.DAM import Seq2Seq2
denoised_model2 = Seq2Seq2()
denoised_model2.load_state_dict(torch.load('./model_save/Seq2Seq2/Seq2Seq2_99_emb_intensity-4.pth'))
denoised_model2 = denoised_model2.cuda()

from model.transformer import ralenet
denoised_model3 = ralenet(high_level_enhence=True)
denoised_model3.load_state_dict(torch.load('./model_save/ralenet/ralenet_99_emb_intensity-4.pth'))
denoised_model3 = denoised_model3.cuda()

with torch.no_grad():
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('test acc: ', acc(pred_array, label_array))
    print('test precision: ', precision(pred_array, label_array))
    print('test f1 score: ', f1_score(pred_array, label_array))
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('noised test acc: ', acc(pred_array, label_array))
    print('noised test precision: ', precision(pred_array, label_array))
    print('noised test f1 score: ', f1_score(pred_array, label_array))
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        data = denoised_model1(data)
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('noised1 test acc: ', acc(pred_array, label_array))
    print('noised1 test precision: ', precision(pred_array, label_array))
    print('noised1 test f1 score: ', f1_score(pred_array, label_array))
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        data = denoised_model2(data)
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('noised2 test acc: ', acc(pred_array, label_array))
    print('noised2 test precision: ', precision(pred_array, label_array))
    print('noised2 test f1 score: ', f1_score(pred_array, label_array))
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.float().cuda()
        label = label.long().cuda()
        data = denoised_model3(data)
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('noised3 test acc: ', acc(pred_array, label_array))
    print('noised3 test precision: ', precision(pred_array, label_array))
    print('noised3 test f1 score: ', f1_score(pred_array, label_array))
    pred_array = []
    label_array = []
    for _, (data, label) in enumerate(noised_test_dataloader):
        data = data.numpy()
        label = label.long().cuda()
        from local_utils.denoisefunc import wavelet_denoise
        data = wavelet_denoise(data)
        data = torch.FloatTensor(data).cuda()
        pred = model(data)
        pred_array.append(pred)
        label_array.append(label)
    pred_array = torch.cat(pred_array, dim=0)
    label_array = torch.cat(label_array, dim=0)
    print('dwt test acc: ', acc(pred_array, label_array))
    print('dwt test precision: ', precision(pred_array, label_array))
    print('dwt test f1 score: ', f1_score(pred_array, label_array))
    

