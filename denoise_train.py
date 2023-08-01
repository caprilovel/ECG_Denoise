import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat

from global_utils.torch_utils.log_utils import train_log
from datetime import datetime


@train_log()
def train(epochs, model, batch_size, train_loader, test_loader, use_gpu,model_path=None, *args, **kwargs):
    from tqdm import tqdm
    if model_path:
        model = model.load_state_dict(torch.load(model_path))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss_list = []
    eval_loss_list = []
    train_nums = len(train_loader.dataset)
    test_nums = len(test_loader.dataset)
    
    init_time = datetime.now()
    for epoch in range(epochs):
        train_snr = []
        train_rmse = []
        test_snr = []
        test_rmse = [] 
        with tqdm(total=(train_nums-1)//batch_size + 1, bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',) as t:
            epoch_init_time = datetime.now()
            
            model.train()
            epoch_loss_list = []
            
            for _, (data, target) in enumerate(train_loader):
                if use_gpu:
                    data, target = data.float().cuda(), target.float32().cuda()
                    
                optimizer.zero_grad()
                pre = model(data)
                loss = F.mse_loss(pre, target)
                epoch_loss_list.append(loss.item())
                
                loss.backward()
                optimizer.step()
                total_time = datetime.now() - init_time
                epoch_time = datetime.now() - epoch_init_time
                t.update(1)
                t.set_description_str(f"\33[36m【Train Epoch {epoch + 1:03d}】") # 设置迭代数描述字符串
                t.set_postfix_str(f"epoch_train_loss={epoch_loss_list[-1] / batch_size:.4f}, epoch_time:{epoch_time}, total_time:{total_time} ") 
        
        model.eval()
        with tqdm(total=(test_nums-1)//batch_size + 1, bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',) as t:
            epoch_eval_loss_list = []
            for _, (data, target) in enumerate(test_loader):
                if use_gpu:
                    data, target = data.float().cuda(), target.float32().cuda()
                pre = model(data)
                loss = F.mse_loss(pre, target)
                epoch_eval_loss_list.append(loss.item())
                total_time = datetime.now() - init_time
                epoch_time = datetime.now() - epoch_init_time
                t.update(1)
                t.set_description_str(f"\33[36m【Eval Epoch {epoch + 1:03d}】")
                t.set_postfix_str(f"epoch_train_loss={epoch_loss_list[-1] / batch_size:.4f}, epoch_time:{epoch_time}, total_time:{total_time} ") 