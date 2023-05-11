#--------------------------------------#

#   显卡设置

#--------------------------------------#

import sys,os
sys.path.append('..')
from global_utils.find_gpus import find_gpus
os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus()

#--------------------------------------#

#   import设置

#--------------------------------------#
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from global_utils.utils import get_time_str,mkdir,Logger,boolean_string

import time 
import wfdb
import sklearn
import scipy
import sktime 
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

#--------------------------------------#

#   命令行参数设置

#--------------------------------------#

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100, help="random_seed")
parser.add_argument('--batch_size', type=int, default=32, help="batch size")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-5, help="weight decay")
parser.add_argument('--epochs', type=int, default=100, help="epochs")
parser.add_argument('--use_arg', type=boolean_string, default=False, help='whether use the args')
parser.add_argument('--use_log', type=boolean_string, default=False, help='whether use the log')


args = parser.parse_args()
if args.use_arg:
    seed = args.seed
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    epochs = args.epochs
    use_log = args.use_log
else:
    seed = 100
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-5
    epochs = 100
    use_log = True


#--------------------------------------#

#   日志设置

#--------------------------------------#
from global_utils.utils import get_time_str,mkdir,Logger,boolean_string
mkdir('./log')
if use_log:
    sys.stdout = Logger('./log/{}log.log'.format(get_time_str()))
    print("-------------Generate Log File---------------")
    print('log file dir:./log/')


#--------------------------------------#

#   随机种子设置

#--------------------------------------#
from global_utils.utils import random_seed
seed = args.seed 
random_seed(seed)

print("\n\n\n-------------Random Seed---------------\n")
print(f"seed is {seed}")



data_path = "../data/mit-bih-arrhythmia-database-1.0.0/"
noise_path = "../data/mit-bih-noise-stress-test-database-1.0.0/"
arryth_path = "../data/mit-bih-arrhythmia-database-1.0.0/"

from ECGDenoise.local_utils.local_utils import batch_arrythdata_iter, Seq2Seq2

import torch.optim as optim
train_num = '118'
train_db = '24'
dataset_divide = [0.6, 0.2, 0.2]
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
Total_DS = DS1 + DS2
random.shuffle(Total_DS)
DS_train = Total_DS[0:int(dataset_divide[0] * len(Total_DS))]
DS_eval = Total_DS[int(dataset_divide[0] * len(Total_DS)):int((dataset_divide[1] + dataset_divide[0]) * len(Total_DS))]
DS_test = Total_DS[int((dataset_divide[1] + dataset_divide[0]) * len(Total_DS)):]    
save_path = "./model_data_pth.tar"
import time 
model = Seq2Seq2()
from global_utils.utils import second2time
from datetime import datetime
model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
mkdir('./model_data/')
def train(epochs, model, batch_size=batch_size, use_gpu=True):
    train_loss_list = [sys.maxsize]
    eval_loss_list = [sys.maxsize]
    init_time = datetime.now()
    print("-------------Init Training!---------------")
    for epoch in range(epochs):
        with tqdm(total=int(650000/(256*batch_size)) * len(DS_train),bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',) as t:
            epoch_init_time = datetime.now()
            
            model.train()
            epoch_loss_list = []
            
            for inputs, labels in batch_arrythdata_iter(DS_train, batch_size):
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                pre = model(inputs)
                loss = F.mse_loss(pre, labels)
                epoch_loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
                total_time = datetime.now() - init_time
                delta_time = datetime.now() - epoch_init_time
                t.update(1)            
                t.set_description_str(f"\33[36m【Train Epoch {epoch + 1:03d}】") # 设置迭代数描述字符串
                t.set_postfix_str(f"epoch_train_loss={epoch_loss_list[-1] / batch_size:.6f}, epoch_time:{delta_time}, total_time:{total_time} ") # 设置后缀描述字符串 
                
            
        model.eval()
        with tqdm(total=int(650000/(256*batch_size)) * len(DS_eval),bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',) as t:
            epoch_init_time = datetime.now()
            
            model.train()
            epoch_loss_list = []
            
            for inputs, labels in batch_arrythdata_iter(DS_eval, batch_size):
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                pre = model(inputs)
                loss = F.mse_loss(pre, labels)
                epoch_loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
                total_time = datetime.now() - init_time
                delta_time = datetime.now() - epoch_init_time
                t.update(1)            
                t.set_description_str(f"\33[36m【Eval  Epoch {epoch + 1:03d}】") # 设置迭代数描述字符串
                t.set_postfix_str(f"epoch_train_loss={epoch_loss_list[-1] / batch_size:.6f}, epoch_time:{delta_time}, total_time:{total_time} ") # 设置后缀描述字符串 
            
        if epoch%5 == 0:
            torch.save(model, './model_data/model_{}_epoch_{}'.format(type(model).__name__, epoch))
    
    # print("init train!")
    # if use_gpu:
    #     model = model.cuda()
    # ttime = time.time()
    # model.train()
    # loss_list = []
    # eval_list = []
    # for epoch in range(epochs):
    #     loss_total = []
    #     for inputs,outputs in batch_arrythdata_iter(DS_train, batch_size):
    #         if use_gpu:
    #             inputs,outputs = inputs.cuda(), outputs.cuda()
    #         # print(inputs.shape)
    #         optimizer.zero_grad()
    #         pre = model(inputs)
    #         # print(pre.shape)
    #         loss = F.mse_loss(pre, outputs)
    #         loss_total.append(loss.item())
    #         loss.backward()
    #         optimizer.step()
    #     loss_list.append(sum(loss_total))
    #     if epoch % 5 == 0:
    #         torch.save(model, "./model_data_epoch{}.pth".format(epoch))
    #     model.eval()
        
    #     eval_mse = []
    #     for inputs,outputs in batch_arrythdata_iter(DS_eval, batch_size):
    #         if use_gpu:
    #             inputs,outputs = inputs.cuda(), outputs.cuda()
            
    #         pre = model(inputs)
    #         loss = F.mse_loss(pre, outputs)
    #         eval_mse.append(loss.item())
    #     eval_mse_value = sum(eval_mse)
    #     print(eval_mse_value)
    #     eval_list.append(eval_mse_value)
    # return eval_list

train(100, model)

