
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
    # change the data into FloatTensor
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
from local_utils.evaluate import RMSE, SNR
from local_utils.denoisefunc import wavelet_denoise
rmse_array = []
snr_array = []
for _, (inputs, targets) in enumerate(train_loader):
    # print(inputs.shape)
    # print(targets.shape)
    input_np = inputs.numpy()
    # target_np = targets.numpy()
    out = wavelet_denoise(input_np)
    out = torch.FloatTensor(out)
    # print(RMSE(out, targets), SNR(out, targets))
    rmse_array.append(RMSE(out, targets))
    snr_array.append(SNR(out, targets))
    
rmse_array = torch.concat(rmse_array, dim=0)
snr_array = torch.concat(snr_array, dim=0)
print("noise_name:", noise_name, "noise_intensity:", noise_intensity)
print("rmse:",rmse_array.mean(),"snr:", snr_array.mean())
with open("dwt_dn_result.txt", "a") as f:
    f.write("noise_name:"+noise_name+"noise_intensity:"+str(noise_intensity)+"rmse:"+str(rmse_array.mean())+"snr:"+str(snr_array.mean())+"\n")