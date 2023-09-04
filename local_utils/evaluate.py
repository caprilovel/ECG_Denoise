import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat


def RMSE(y, y_pred):
    """rooted mean squared error

    Args:
        y (tensor): origin data, Batch x Channel x Length
        y_pred (tensor): reconstructed data. Batch x Channel x Length
    
    Returns:
        RMSE (tensor): RMSE value, Batch
        
    Examples:
        >>> a = torch.randn(2, 3, 4)
        >>> b = torch.randn(2, 3, 4)
        >>> c = RMSE(a, b)
        >>> c.shape
        (2,)
    """
    y = y.flatten(start_dim=1)
    y_pred = y_pred.flatten(start_dim=1)
    return torch.sqrt(torch.mean((y - y_pred) ** 2, dim=-1))

def SNR(y, y_pred):
    """signal to noise ratio

    Args:
        y (tensor): origin data, Batch x Channel x Length
        y_pred (tensor): reconstructed data. Batch x Channel x Length
        
    Returns:
        SNR (tensor): SNR value, Batch
    
    Examples:
        >>> a = torch.randn(2, 3, 4)
        >>> b = torch.randn(2, 3, 4)
        >>> c = RMSE(a, b)
        >>> c.shape
        (2,)
        
    """
    y = y.flatten(start_dim=1)
    y_pred = y_pred.flatten(start_dim=1)
    return 10 * torch.log10(torch.mean(y ** 2, dim=-1) / torch.mean((y - y_pred) ** 2, dim=-1))

if __name__ == "__main__":
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 3, 4)
    a = a.detach()
    print(SNR(a, b))
    a = a * 10 
    b = b * 10 
    print(SNR(a, b))


