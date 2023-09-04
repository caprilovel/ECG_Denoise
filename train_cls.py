import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat

from torch.utils.data import Dataset, DataLoader

from model.ResNet_cls import ResNet_cls

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
    
train_dataset = ClsDataset(train=True)
test_dataset = ClsDataset(train=False)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)     
def acc(pred, label):
    pred = torch.argmax(pred, dim=1)
    return torch.sum(pred == label).item() / len(label)
model = ResNet_cls()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model = model.cuda()
for epoch in range(20):
    epoch_pred = []
    epoch_label = []
    for _, (data, label) in enumerate(train_dataloader):
        # print(data.shape, type(data), label.shape, type(label))
        data = data.float().cuda()
        label = label.long().cuda()
        pred = model(data)
        loss = F.cross_entropy(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_pred.append(pred)
        epoch_label.append(label)
    epoch_pred = torch.cat(epoch_pred, dim=0)
    epoch_label = torch.cat(epoch_label, dim=0)
    train_acc = acc(epoch_pred, epoch_label)
    print('epoch: {}, train_acc: {}'.format(epoch, train_acc))

torch.save(model.state_dict(), './model/cls_model.pth')

