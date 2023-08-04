import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat

from global_utils.torch_utils.log_utils import train_log
from datetime import datetime
from local_utils.evaluate import RMSE, SNR


@train_log()
def train(epochs, model, batch_size, train_loader, test_loader, use_gpu,model_path=None, *args, **kwargs):
    for k,v in kwargs.items():
        print(k,":", v)
    if use_gpu:
        model = model.cuda()
    from tqdm import tqdm
    if model_path:
        model = model.load_state_dict(torch.load(model_path))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss_list = []
    eval_loss_list = []
    train_snr_list = []
    test_snr_list = []
    train_rmse_list = []
    test_rmse_list = []
    
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
                    data, target = data.cuda(), target.cuda()
                    
                optimizer.zero_grad()
                pre = model(data)
                loss = F.mse_loss(pre, target)
                epoch_loss_list.append(loss.item())
                
                loss.backward()
                optimizer.step()
                train_snr.append(SNR(target, pre))
                train_rmse.append(RMSE(target, pre))
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
                    data, target = data.cuda(), target.cuda()
                pre = model(data)
                loss = F.mse_loss(pre, target)
                epoch_eval_loss_list.append(loss.item())
                total_time = datetime.now() - init_time
                epoch_time = datetime.now() - epoch_init_time
                test_snr.append(SNR(target, pre))
                test_rmse.append(RMSE(target, pre))
                t.update(1)
                t.set_description_str(f"\33[36m【Eval Epoch {epoch + 1:03d}】")
                t.set_postfix_str(f"epoch_train_loss={epoch_loss_list[-1] / batch_size:.4f}, epoch_time:{epoch_time}, total_time:{total_time} ") 
        train_snr = torch.cat(train_snr, dim=0)
        train_rmse = torch.cat(train_rmse, dim=0)
        test_snr = torch.cat(test_snr, dim=0)
        test_rmse = torch.cat(test_rmse, dim=0)
        train_snr_list.append(train_snr.mean().item())
        test_snr_list.append(test_snr.mean().item())
        train_rmse_list.append(train_rmse.mean().item())
        test_rmse_list.append(test_rmse.mean().item())
        if (epoch + 1) % 10 == 0:
            from global_utils.torch_utils.log_utils import mkdir
            mkdir("model_save")
            torch.save(model.state_dict(), f"model_save/{model.__class__.__name__}_{epoch}.pth")
            print(f"model_save/{model.__class__.__name__}_{epoch}_{kwargs['noise_name']}_intensity{kwargs['noise_intensity']}.pth")
            print("epoch:", epoch + 1)
            print("train snr:", train_snr.mean().item())
            print("test snr:", test_snr.mean().item())
            print("train rmse:", train_rmse.mean().item())
            print("test rmse:", test_rmse.mean().item())
    with open("./output.txt", "a") as output_file:
        output_file.write(f"{kwargs['model_name']}_{epoch}_{kwargs['noise_name']}_intensity{kwargs['noise_intensity']}:snr:{test_snr.mean().item()}, rmse:{test_rmse.mean().item()}\n") 
        
    return train_snr_list, test_snr_list, train_rmse_list, test_rmse_list