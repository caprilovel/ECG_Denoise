import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat


from model_transformer import * 


class PatchEmbedding1d(nn.Module):
    '''
    Sequence to Patch Embedding
    
    Args:
        patch_size (int): patch size
        in_channels (int): input channels
        embed_dim (int): embedding dimension
        norm_layer (nn.Module, optional): normalization layer. Defaults to None.
        
    Examples:
        >>> x = torch.randn(1, 12, 1000)
        >>> patch_embedding = PatchEmbedding1d(patch_size=10, in_channels=12, embed_dim=64)
        >>> x = patch_embedding(x)
        >>> x.shape
        torch.Size([1, 100, 64])
    
    '''
    def __init__(self, patch_size, in_channels, embed_dim, norm_layer =None):
        '''
        Sequence to Patch Embedding
            
            Args:
                patch_size (int): patch size
                in_channels (int): input channels
                embed_dim (int): embedding dimension
                norm_layer (nn.Module, optional): normalization layer. Defaults to None.
                
            Examples:
                >>> x = torch.randn(1, 12, 1000)
                >>> patch_embedding = PatchEmbedding1d(patch_size=10, in_channels=12, embed_dim=64)
                >>> x = patch_embedding(x)
                >>> x.shape
                torch.Size([1, 100, 64])    
        '''
        super(PatchEmbedding1d, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim 
        
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        #todo using stride=1 and a linear project to generate the embedding vector
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        B, C, L = x.shape
         
        x = rearrange(self.conv(x), 'b c l -> b l c')
        if self.norm:
            x = self.norm(x)
        return x

class PatchMerging1d(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm) -> None:
        """_summary_

        Args:
            dim (_type_): _description_
            out_dim (_type_, optional): _description_. Defaults to None.
            norm_layer (_type_, optional): _description_. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim else dim * 2
        self.norm = norm_layer(self.out_dim)
        
    def forward(self, x):
        B, L, C = x.shape
        x = rearrange(x, 'b (h w) c -> b (c w) h ', w=2)
        x = self.norm(x)
        return x

        
        
        

class SwinTransformer1d(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, ) -> None:
        super().__init__()
        
        
        
    def forward(self, x):
        pass

if __name__ == "__main__":
    x = torch.randn(1, 12, 1000)
    patch_embedding = PatchEmbedding1d(patch_size=10, in_channels=12, embed_dim=64)
    x = patch_embedding(x)
    print(x.shape)
    