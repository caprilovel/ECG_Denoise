from global_utils.torch_utils.cuda import find_gpus
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat

from global_utils.torch_utils.log_utils import Logger, easymail
from global_utils.torch_utils.torch_utils import random_seed
from global_utils.torch_utils.Args import TorchArgs
from local_utils.data_utils import Ecg_Dataset
import random
from torch.utils.data import DataLoader, Dataset, random_split, Subset

import warnings
warnings.filterwarnings('ignore')

random_seed(2023)
noise_intensities = [-4, -2, 0, 2, 4]
noise_type = ['bw', 'ma', 'em', 'emb']
models = ['unet', 'DANet', "ralenet", "testmodel"]

args = TorchArgs()
args.add_argument("--intensity_index", type=int, default=0)
args.add_argument("--noise_type_index", type=int, default=0)
args.add_argument("--model_index", type=int, default=0)


args_dict = vars(args.parse_args())
for k,v in args_dict.items():
    print(k, v)

batch_size = args_dict['batch_size']    
noise_name = noise_type[args_dict['noise_type_index']]
noise_intensity = noise_intensities[args_dict['intensity_index']]
ecg_data = Ecg_Dataset(noise_name=noise_name, noise_intensity=noise_intensity)

def custom_collate_fn(batch):
    # 将数据转换为FloatTensor类型
    inputs, targets = zip(*batch)
    return torch.FloatTensor(inputs), torch.FloatTensor(targets)
total_samples = len(ecg_data)
select_samples = random.sample(range(total_samples), 10000)
select_dataset = Subset(ecg_data, select_samples)
train_ratio = 0.8
test_ratio = 0.2 
train_size = int(train_ratio * len(select_dataset))
test_size = len(select_dataset) - train_size
train_dataset, test_dataset = random_split(select_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True, collate_fn=custom_collate_fn)

from denoise_train import train 
if args_dict['model_index'] == 0:
    from model.UNet import UNet
    model = UNet()
    model_name = 'unet'
if args_dict['model_index'] == 1:
    from model.DAM import Seq2Seq2
    model = Seq2Seq2()
    model_name = 'DANet'
if args_dict['model_index'] == 2:
    from model.raletransformer import ralenet
    model = ralenet()
    model_name = 'ralenet'    

if args_dict['model_index'] == 3:
    from model.transformer import ralenet
    model = ralenet(high_level_enhence=True)
    model_name = 'testmodel'
    
if args_dict['model_index'] == 4:
    from model.transformer import ralenet
    model = ralenet(low_level_enhence=False)
    model_name = 'modelmlp'



epochs = args_dict['epochs']
batch_size = args_dict['batch_size']
train(epochs=epochs, batch_size=batch_size, model=model, train_loader=train_loader, test_loader=test_loader, model_name=model_name, use_gpu=True, noise_name=noise_name, noise_intensity=noise_intensity,)

# if __name__ == "__main__":
#     x = torch.randn(32, 2, 256)
#     for _, (x, y) in enumerate(train_loader):
#         print(x.shape)
#         z = model(x)
#         print(z.shape)

