import sys,os
sys.path.append('..')
#--------------------------------------#

#   命令行参数设置

#--------------------------------------#
import argparse
from global_utils.log_utils import boolean_string, timestamp
from global_utils.mailsend import EmailSender


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100, help="random_seed")
parser.add_argument('--batch_size', type=int, default=32, help="batch size")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-5, help="weight decay")
parser.add_argument('--epochs', type=int, default=100, help="epochs")
# parser.add_argument('--use_arg', type=boolean_string, default=False, help='whether use the args')
parser.add_argument('--use_log', type=boolean_string, default=True, help='whether use the log')
parser.add_argument('--load_model', type=boolean_string, default=False, help='whether read the model')
parser.add_argument('--load_epoch', type=int, default=20, help='the epoch to be read')
parser.add_argument('--augment', type=str, default='none', help='')
parser.add_argument('--snr', type=float, default=0, help='')
parser.add_argument('--noise_type', type=str, default='em', help='')
parser.add_argument('--use_smooth_loss', type=boolean_string, default=True, help='')

parser.add_argument('--qkv_proj', type=str, default='linear', help='qkv projection type')
parser.add_argument('--ffn_type', type=str, default='leff', help='feed forward network type')
# from torch.utils.tensorboard import SummaryWriter

use_arg = True 
args = parser.parse_args()
if use_arg:
    seed = args.seed
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    epochs = args.epochs
    use_log = args.use_log
    load_model = args.load_model
    load_epoch = args.load_epoch
    snr = args.snr 
    noise_type = args.noise_type
    qkv_proj = (args.qkv_proj).lower()
    ffn_type = (args.ffn_type).lower()
    use_smooth_loss = args.use_smooth_loss
    
else:
    seed = 100
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-5
    epochs = 100
    use_log = True
    load_model = False 
    load_epoch = -1
    snr = 0 
    noise_type = 'em'
    
    qkv_proj = 'linear'
    ffn_type = 'mlp'
    use_smooth_loss = True


#--------------------------------------#

#   日志设置

#--------------------------------------#
from global_utils.log_utils import get_time_str,mkdir,Logger

mkdir('./log')
if use_log:
    log_path = './log/{}log.txt'.format(get_time_str())
    sys.stdout = logger = Logger(log_path)
    print("-------------Generate Log File---------------")
    print('log file dir:./log/')
    print('log: {}'.format(log_path))

print("-------------Arguments---------------")
for arg in vars(args):
    print(arg, ':', getattr(args, arg))

    
#--------------------------------------#

#   显卡设置

#--------------------------------------#
from global_utils.find_gpus import find_gpus

print("-------------GPUs Distribution---------------")
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
from datetime import datetime
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

#   路径设置

#--------------------------------------#

noise_path = "../data/mit-bih-noise-stress-test-database-1.0.0/"
arryth_path = "../data/mit-bih-arrhythmia-database-1.0.0/"
model_data_path = './model_data/'
model_search_path = './model_data/model*'

#--------------------------------------#

#   随机种子设置

#--------------------------------------#
from global_utils.utils import random_seed

random_seed(seed)
print("\n\n\n-------------Random Seed---------------\n")
print(f"seed is {seed}")


#--------------------------------------#

#   数据集划分设置

#--------------------------------------#
from local_utils.local_utils import batch_arrythdata_iter, Seq2Seq2, batch_norm_arrythdata_iter, batch_norm_snr_iter

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
print('TrainSet: ', DS_train, f'num: {len(DS_train)}')
print('EvalSet:  ', DS_eval, f'num: {len(DS_eval)}')
print('TestSet:  ', DS_test, f'num: {len(DS_test)}')

#--------------------------------------#

#   超参数设置

#--------------------------------------#
use_gpu = True
from model.model_transformer import Transformer
# 设置模型保存参数规格
init_epoch = 0

model = Transformer(token_projection=qkv_proj, token_mlp=ffn_type)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
mkdir('./model_data/')
print()


#--------------------------------------#

#   模型读取设置

#--------------------------------------#
from local_utils.local_utils import find_model, second_order_difference_loss
t_stamp = timestamp()
mkdir(f'./model_data/{type(model).__name__}/')
mkdir(f'./model_data/{type(model).__name__}/{t_stamp}/')
with open(f'./model_data/{type(model).__name__}/{t_stamp}/config.txt', 'w') as f:
    for arg in vars(args):
        f.write(f'{arg}: {getattr(args, arg)}\n')
    f.write('DS_train:' + str(DS_train) + '\n')
    f.write('DS_eval:' + str(DS_eval) + '\n')
    f.write('DS_test:' + str(DS_test) + '\n')
print("\n-------------Load Model---------------")
if load_model:
    if find_model(model_search_path, type(model).__name__, epoch=load_epoch):
        print('load model path: ', find_model(model_search_path, type(model).__name__, epoch=load_epoch))
        model = torch.load(find_model(model_search_path, type(model).__name__, epoch=load_epoch))
        init_epoch = int(find_model(model_search_path, type(model).__name__, epoch=load_epoch).split('_')[-1].split('.')[0])
    
    else:
        print('model data doesn\'t exist, training from the ground up!\n')
else:
    print('\n training from the ground up!')
    
if use_gpu:
    model = model.cuda()  
    




#--------------------------------------#

#   定义训练函数以及测试函数

#--------------------------------------#


print('save_model_path: ./model_data/m_{}_{}_epoch_X.pth'.format(t_stamp,type(model).__name__))

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
            
            for inputs, labels in batch_norm_snr_iter(DS_train, batch_size, snr=snr, noise_type=noise_type):
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    # size: batch x channels x length
                optimizer.zero_grad()
                pre = model(inputs)
                loss = F.mse_loss(pre, labels)
                if use_smooth_loss:
                    loss += second_order_difference_loss(pre, dim=2) * 0.01
                epoch_loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
                total_time = datetime.now() - init_time
                delta_time = datetime.now() - epoch_init_time
                t.update(1)            
                t.set_description_str(f"\33[36m【Train Epoch {epoch + 1:03d}】") # 设置迭代数描述字符串
                t.set_postfix_str(f"epoch_train_loss={epoch_loss_list[-1] / batch_size:.4f}, epoch_time:{delta_time}, total_time:{total_time} ") # 设置后缀描述字符串 
                
            
        model.eval()
        with tqdm(total=int(650000/(256*batch_size)) * len(DS_eval),bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',) as t:
            epoch_init_time = datetime.now()
            
            epoch_loss_list = []
            
            for inputs, labels in batch_norm_snr_iter(DS_eval, batch_size, snr=snr, noise_type=noise_type):
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                pre = model(inputs)
                loss = F.mse_loss(pre, labels)
                epoch_loss_list.append(loss.item())
                total_time = datetime.now() - init_time
                delta_time = datetime.now() - epoch_init_time
                t.update(1)            
                t.set_description_str(f"\33[36m【Eval  Epoch {epoch + 1:03d}】") # 设置迭代数描述字符串
                t.set_postfix_str(f"epoch_eval_loss={epoch_loss_list[-1] / batch_size:.4f}, epoch_time:{delta_time}, total_time:{total_time} ") # 设置后缀描述字符串 
            
        if (epoch+1) % 5 == 0:
            mkdir(f'./model_data/{type(model).__name__}/')
            mkdir(f'./model_data/{type(model).__name__}/{t_stamp}/')
            # model_data_dir = f'./model_data/m_{timestamp()}_{type(model).__name__}_epoch_*.pth'
            # torch.save(model, './model_data/m_{}_{}_epoch_{}.pth'.format(t_stamp,type(model).__name__, epoch + 1))
            torch.save(model.state_dict(), './model_data/{}/{}/m_{}_{}_epoch_{}.pth'.format(type(model).__name__, t_stamp, t_stamp,type(model).__name__, epoch + 1))

    
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

from local_utils.local_utils import torch_calcu_snr
def test():
    epoch_loss_list = []
    snr_loss_list = []
            
    for inputs, labels in batch_norm_snr_iter(DS_test, batch_size, snr=snr, noise_type=noise_type):
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        pre = model(inputs)
        loss = F.mse_loss(pre, labels)
        snr = torch_calcu_snr(pre, labels)
        epoch_loss_list.append(loss.item())
        snr_loss_list.append(snr)
    
    epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
    snr_mean = sum(snr_loss_list) / len(snr_loss_list)
    

#--------------------------------------#

#   训练结束发送邮件

#--------------------------------------#

print("-------------training complete!---------------")



print("-------------!---------------")
logger.close()
es = EmailSender()
es.get_attachment(log_path)
es.get_text('training complete!')
es.send()