import os

import numpy as np
import pandas as pds
from torch.utils.data import Dataset
from einops import reduce, rearrange


class Normalizer(object):
    """Normalizer _summary_

    Used for the data normalization

    """
    def __init__(self,
                 norm_type,
                 mean=None,
                 std=None,
                 min_val=None,
                 max_val=None,
                 eps=None) -> None:
        """__init__ _summary_

        _extended_summary_

        Args:
            norm_type (_type_): _description_
            mean (_type_, optional): _description_. Defaults to None.
            std (_type_, optional): _description_. Defaults to None.
            min_val (_type_, optional): _description_. Defaults to None.
            max_val (_type_, optional): _description_. Defaults to None.
            eps (_type_, optional): _description_. Defaults to None.
        """
        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        

    def normalize(self, df):
        """normalize _summary_

        _extended_summary_

        Args:
            df (numpy.array or torch.Tensor): should be shape like batch * channels * length or channels * length. Anyway, the last dim should be the length of the dataframe.
        """
        # _type = None
        # from numpy import ndarray
        # from torch import Tensor
        # if type(df) == ndarray:
        #     _type = "ndarray"
        # elif type(df) == Tensor:
        #     _type = "tensor"
        
        if self.norm_type == " standardization":
            if self.mean is None:
                self.mean = reduce(df, "... l -> ... 1", 'mean')
                self.std = 
            return (df - self.mean) / (self.std + self.eps)
        
class ClassifyDataset(Dataset):
    def __init__(self, data, indices, ):
        self.data = data
        self.IDs = indices
        
        
        
                
    def __getitem__(self, index):
        """__getitem__ _summary_

        _extended_summary_

        Args:
            index (ndarray or List): the index of samples in dataset  

        Returns:
            _type_: _description_
        """
        
        
        return self.data[index]
    
    def __len__(self):
        return len(self.IDs)