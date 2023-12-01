import random
from einops import rearrange, reduce
import wfdb
import numpy as np
import torch 
import torch.nn as nn
import glob
import math

noise_path = "../data/mit-bih-noise-stress-test-database-1.0.0/"
arryth_path = "../data/mit-bih-arrhythmia-database-1.0.0/"




path = "../data/mit-bih-noise-stress-test-database-1.0.0/"


def ecg_noise_add(data, alpha=0.1, random_seed=500):
    random.seed(random_seed)
    datashape = data.shape
    noise_index = ['bw', 'ma', 'em']
    length = data.shape[-2]
    if len(data.shape) is 2:
        data = np.expand_dims(data, 0)
    noise_list = []
    for i in range(data.shape[0]):
        group_index = random.randint(0, len(noise_index) - 1)
        data_index = random.randint(0, 650000 - int(data.shape[-2]) - 1)
        noise_list.append(
            alpha *
            wfdb.rdrecord(path + noise_index[group_index],
                          physical=False).d_signal[data_index:data_index +
                                                   int(data.shape[-2])])
    output = data + np.stack(noise_list, 0)
    if output.shape != datashape:
        output = np.squeeze(output, 0)
    return output


data_path = "../data/mit-bih-arrhythmia-database-1.0.0/"

noise_path = "../data/mit-bih-noise-stress-test-database-1.0.0/"
arryth_path = "../data/mit-bih-arrhythmia-database-1.0.0/"


def batch_data_iter(num, db, batch_size):
    signal = wfdb.rdrecord(noise_path + '118e24', physical=False).d_signal
    true_sig = wfdb.rdrecord(noise_path + num + 'e00', physical=False).d_signal
    flag = 0
    data_sig = []
    data_true = []
    for i in range(0, 650000, 256):
        if flag < batch_size:
            data_sig.append(torch.FloatTensor(signal[i:i + 256, :]))
            data_true.append(torch.FloatTensor(true_sig[i:i + 256, :]))
            flag += 1
        else:
            yield torch.transpose(torch.stack(data_sig, dim=0), 1,
                                  2).contiguous(), torch.transpose(
                                      torch.stack(data_true, dim=0), 1,
                                      2).contiguous()
            data_sig = []
            data_true = []
            flag = 0


def batch_arrythdata_iter(nums, batch_size):
    for num in nums:
        signal = wfdb.rdrecord(arryth_path + str(num), physical=False).d_signal
        flag = 0
        data_sig = []
        for i in range(0, 650000, 256):
            if flag < batch_size:
                data_sig.append(signal[i:i + 256, :])
                flag += 1
            else:
                ground_truth_data = np.stack(data_sig, 0)
                yield torch.transpose(
                    torch.FloatTensor(ecg_noise_add(ground_truth_data)), 1,
                    2).contiguous(), torch.transpose(
                        torch.FloatTensor(ground_truth_data), 1,
                        2).contiguous()
                data_sig = []
                flag = 0
def Gnoisegen(x, y, snr):
    '''
    Generates a noisy signal based on the snr
    
    Args:
        x: numpy.array. Clean signal. 
        y: numpy.array. Noise signal which would be add to the clean signal.
        snr: float. Signal noise ratio.
        
    Examples:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.random.randn(5)
        >>> snr = 10
        >>> z = Gnoisegen(x, y, snr)
        >>> z

        
    '''
    x_temp = x.flatten()
    
    Nx = len(x)  # 求出信号的长度
    
    noise = y.flatten()
    signal_power = np.sum(x_temp*x_temp)/Nx# 求信号的平均能量
    noise_power = np.sum(noise*noise)/Nx# 求信号的平均能量
    noise_variance = signal_power/(math.pow(10., (snr/10)))#计算噪声设定的方差值
    noise = math.sqrt(noise_variance/noise_power)*y# 按照噪声能量构成相应的白噪声
    z=x+noise
    return(z, noise)

def batch_norm_snr_iter(nums, batch_size, snr, noise_type):
    noise_data = wfdb.rdrecord(noise_path + noise_type, physical=False).d_signal
    for num in nums:
        signal = wfdb.rdrecord(arryth_path + str(num), physical=False).d_signal
        
        for i in range(0, 650000, 256*batch_size):
            if i+256*batch_size>650000:
                break
            data_sig = signal[i:i+256*batch_size, :]
            data_sig = np_norm(data_sig, dim=0)
            data_index = random.randint(0, 650000-int(data_sig.shape[-2])-1)
            noisy_data = Gnoisegen(data_sig, noise_data[data_index:data_index+int(data_sig.shape[-2]), :], snr)[0]
            noisy_data = torch.FloatTensor(noisy_data)
            data_sig = torch.FloatTensor(data_sig)
            yield rearrange(noisy_data, '(b l) c -> b c l', b=batch_size), rearrange(data_sig, '(b l) c -> b c l', b=batch_size) 



def torch_calcu_snr(noisy_data, data):
    if len(noisy_data.shape) == 2:
        noisy_data = torch.unsqueeze(noisy_data, 0)
        data = torch.unsqueeze(data, 0)
    signal_power = torch.mean(torch.norm(data, dim=(1, 2)) ** 2, dim=0)
    noise_power = torch.mean(torch.norm(data - noisy_data, dim=(1, 2)) ** 2, dim=0)
    return 10 * torch.log10(signal_power / noise_power)
    

def np_calcu_snr(noisy_data, data):
    # noisy_data: (batch, length, channel)
    # data: (batch, length, channel)
    if len(noisy_data.shape) == 2:
        noisy_data = np.expand_dims(noisy_data, 0)
        data = np.expand_dims(data, 0)
    noisy_data = noisy_data.reshape(noisy_data.shape[0], -1)
    data = data.reshape(data.shape[0], -1)
    noise = noisy_data - data
    
    signal_energy = np.mean(np.abs(data)**2, axis=1)
    noise_energy = np.mean(np.abs(noise)**2, axis=1)
    snr = 10 * np.log10(signal_energy / noise_energy)
    return snr    

def ecg_noise_add(data, alpha=0.1 ,random_seed=500):
    path = "../data/mit-bih-noise-stress-test-database-1.0.0/"
    random.seed(random_seed)
    datashape = data.shape
    noise_index = ['bw', 'ma', 'em']
    length = data.shape[-2]
    if len(data.shape) is 2:
        data = np.expand_dims(data, 0)
    noise_list = []
    for i in range(data.shape[0]):
        group_index = random.randint(0, len(noise_index)-1)
        data_index = random.randint(0, 650000-int(data.shape[-2])-1)
        noise_list.append(alpha * wfdb.rdrecord(path + noise_index[group_index], physical=False).d_signal[data_index:data_index+int(data.shape[-2])])
    output = data + np.stack(noise_list, 0)
    if output.shape != datashape:
        output = np.squeeze(output, 0)
    return output

def single_snr_noise_add(data, noise, snr):
    """
    data: 原始数据。
    noise: 噪声数据。
    snr: 信噪比（单位：dB）。
    """
    # 计算信号能量和噪声能量，并计算信噪比对应的噪声能量。
    signal_energy = np.mean(np.abs(data)**2)
    noise_energy = np.mean(np.abs(noise)**2)
    target_noise_energy = signal_energy / (10**(snr/10))

    # 对噪声进行标准化处理，使其达到目标信噪比。
    scale = np.sqrt(target_noise_energy / noise_energy)
    scaled_noise = noise * scale

    # 返回加入噪声后的数据。
    return data + scaled_noise

def batch_ecg_noise_add_snr(data, snr, noise_type):
    # data shape: (batch, length, channel) 
    path = "../data/mit-bih-noise-stress-test-database-1.0.0/"
    datashape = data.shape
    length = data.shape[-2]
    noisy_list = []
    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
    if type(noise_type) == str:
        assert noise_type in ['bw', 'ma', 'em']
        noise = wfdb.rdrecord(path + noise_type, physical=False).d_signal
        
             
    for i in range(data.shape[0]):
        data_index = random.randint(0, 650000-int(length)-1)
        noise_data = noise[data_index:data_index+int(length)]
        noisy_list.append(single_snr_noise_add(data[i], snr, noise_data))  
        
    output = np.stack(noisy_list, 0)
    if output.shape != datashape:
        output = np.squeeze(output, 0)
    return output
  
    
    
    

def batch_data_iter(num, db, batch_size):
    
    signal = wfdb.rdrecord(noise_path+'118e24', physical=False).d_signal
    true_sig = wfdb.rdrecord(noise_path+num+'e00', physical=False).d_signal 
    flag = 0
    data_sig = []
    data_true = []
    for i in range(0, 650000, 256):
        if flag<batch_size:
            data_sig.append(torch.FloatTensor(signal[i:i+256,:]))
            data_true.append(torch.FloatTensor(true_sig[i:i+256,:]))
            flag += 1
        else:
            yield torch.transpose(torch.stack(data_sig, dim=0), 1, 2).contiguous(), torch.transpose(torch.stack(data_true, dim=0), 1, 2).contiguous()
            data_sig = []
            data_true = []
            flag = 0

def batch_arrythdata_iter(nums, batch_size):
    for num in nums:
        signal = wfdb.rdrecord(arryth_path + str(num), physical=False).d_signal
        flag = 0
        data_sig = []
        for i in range(0, 650000, 256):
            if flag<batch_size:
                data_sig.append(signal[i:i+256, :])
                flag += 1
            else:
                ground_truth_data = np.stack(data_sig, 0)
                yield torch.transpose(torch.FloatTensor(ecg_noise_add(ground_truth_data, alpha=1)), 1, 2).contiguous(), torch.transpose(torch.FloatTensor(ground_truth_data), 1, 2).contiguous()
                data_sig = []
                data_sig.append(signal[i:i+256, :])
                flag = 1




def norm_tensor(a, dim=1):
    m = torch.mean(a, dim).unsqueeze(dim)
    return (a-m)/torch.std(a-m, dim).unsqueeze(dim)
def np_norm(a, dim=1):
    m = np.expand_dims(np.mean(a, axis=dim), axis=dim)
    a = a - m
    std = np.std(a, axis=dim)
    std = np.expand_dims(std, axis=dim)
    return a/std

def batch_norm_arrythdata_iter(nums, batch_size, use_random_noise=False, max_noise=1):
    for num in nums:
        signal = wfdb.rdrecord(arryth_path + str(num), physical=False).d_signal
        for i in range(0, 650000, 256*batch_size):
            if i+256*batch_size>650000:
                break
            data_sig = signal[i:i+256*batch_size, :]
            data_sig = np_norm(data_sig, dim=0)
            if use_random_noise:
                alpha = random.random() * max_noise
            else: 
                alpha = 1
            noise_data = torch.FloatTensor(ecg_noise_add(data_sig, alpha=alpha))
            data_sig = torch.FloatTensor(data_sig)
            yield rearrange(noise_data, '(b l) c -> b c l', b=batch_size), rearrange(data_sig, '(b l) c -> b c l', b=batch_size) 



class APReLU(nn.Module):
    def __init__(self, channels):
        super(APReLU, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        fcnList = [nn.Linear(2 * channels, 2 * channels), nn.BatchNorm1d(2 * channels), nn.ReLU(), nn.Linear(2 * channels, channels), nn.BatchNorm1d(channels), nn.Sigmoid()]
        self.fcn = nn.Sequential(*fcnList)
        
    def forward(self, x):
        zerox = torch.zeros_like(x)
        posx = torch.max(x, zerox)
        negx = torch.min(x, zerox)
        
        concatx = torch.concat([self.gap(posx).squeeze(-1),self.gap(negx).squeeze(-1)], dim=1)
        concatx = self.fcn(concatx)
        return posx + concatx.unsqueeze(2) * negx

class EncoderCell(nn.Module):
    def __init__(self, out_channels, kernel_size, padding, stride=2, ):
        super(EncoderCell, self).__init__()
        self.conv = nn.LazyConv1d(out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.aprelu = APReLU(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.aprelu(out)
        out = self.bn(out)
        # print(out.shape)
        return out 
    
    
class DeNoiseEnc(nn.Module):
    def __init__(self,):
        super(DeNoiseEnc, self).__init__()
        self.conv_kernel = [17, 17, 3, 3]
        self.paddingsize = [8, 8, 1, 1]
        self.out_channels = [4, 8, 16, 32]
        self.EncoderList = nn.ModuleList()
        for i in range(4):
            self.EncoderList.add_module("cell{}".format(i), EncoderCell(self.out_channels[i], self.conv_kernel[i], self.paddingsize[i]))
        
        
        
    def forward(self, x):
        out = []
        for cell in self.EncoderList:
            x = cell(x)
            out.append(x)
        return out



class DAM(nn.Module):
    def __init__(self, channels):
        super(DAM, self).__init__()
        # Channel Attention
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        fcnList = [nn.Linear(channels, channels), nn.BatchNorm1d(channels), nn.ReLU(), nn.Linear(channels, channels), nn.BatchNorm1d(channels), nn.Sigmoid()]
        self.fcn1 = nn.Sequential(*fcnList)
        self.fcn2 = nn.Sequential(*fcnList)
        
        # Spatial Attention
        self.cap = nn.AdaptiveAvgPool1d(1)
        self.cmp = nn.AdaptiveMaxPool1d(1)
        self.convsa = nn.Conv1d(2, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x_transpose = x.transpose(1, 2)
        # Channel Attention
        gapx = self.gap(x).squeeze(2)
        gmpx = self.gmp(x).squeeze(2)
        gapx = self.fcn1(gapx)
        gmpx = self.fcn2(gmpx)
        Cattn = self.sigmoid(gapx + gmpx).unsqueeze(-1)
        
        #Spatial Attn
        capx = self.cap(x_transpose).transpose(1,2)
        cmpx = self.cmp(x_transpose).transpose(1,2)
        catcp = torch.cat((capx, cmpx), dim=1)
        Sattn = self.sigmoid(self.convsa(catcp).squeeze(1)).unsqueeze(-2)
        x = Cattn * x
        x = Sattn * x
        return x
    
class DecoderCell(nn.Module):
    def __init__(self, out_channels, kernel_size, padding, stride=2, last=False):
        super(DecoderCell, self).__init__()
        self.last = last
        
        
        self.deconv = nn.LazyConvTranspose1d(out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.aprelu = APReLU(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        
        if last == False:
            self.dam = DAM(out_channels)
    def forward(self, x):
        outx = self.deconv(x)
        outx = self.aprelu(outx)
        outx = self.bn(outx)
        if self.last is not True:
            outx = self.dam(outx)
        # print(outx.shape)
        return outx




class DeNoiseDec(nn.Module):
    def __init__(self,):
        super(DeNoiseDec, self).__init__()
        self.conv_kernel = [4, 4, 18, 18]
        self.paddingsize = [1, 1, 8, 8]
        self.out_channels = [16, 8, 4, 2]
        DecoderList = []
        for i in range(4):
            if i is not 3:
                DecoderList.append(DecoderCell(self.out_channels[i], self.conv_kernel[i], self.paddingsize[i]))
            else:
                DecoderList.append(DecoderCell(self.out_channels[i], self.conv_kernel[i], self.paddingsize[i], last=True))
        self.DecoderList = nn.ModuleList(DecoderList)
        
        
        
    def forward(self, xlist):
        y3 = self.DecoderList[0](xlist[-1])
        y2 = self.DecoderList[1](y3 + xlist[-2])
        y1 = self.DecoderList[2](y2 + xlist[-3])
        y0 = self.DecoderList[3](y1 + xlist[-4])
        
        
        
        return y0

class Seq2Seq2(nn.Module):
    def __init__(self,):
        super(Seq2Seq2, self).__init__()
        self.enc = DeNoiseEnc()
        self.dec = DeNoiseDec()
        
        
    def forward(self, x):
        return self.dec(self.enc(x))     
    
class L1_Charbonnier_loss(nn.Module):
    def __init__(self, eps=1e-6) -> None:
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = eps
        
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
   
# ################# qkv Projection ####################
# class LinearProjection(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0, bias = True) -> None:
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads 
#         self.to_q = nn.Linear(dim, inner_dim, bias=bias)
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
#         self.dim = dim
#         self.inner_dim = inner_dim
        
#     def forward(self, x, attn_kv=None):
#         '''
#         Args:
#           x: Batch x Length x Dim
        
#         Output:
#           q: Batch x heads x length x head_dim
#           k: Batch x heads x length_kv x head_dim
#           v: Batch x heads x length_kv x head_dim
        
#         '''
#         B_, N, C = x.shape
#         if attn_kv is not None:
#             attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
#         else:
#             attn_kv = x
#         N_kv = attn_kv.size(1)
#         q = self.to_q(x).reshape(B_, N, 1, self.heads, self.inner_dim // self.heads).permute(2, 0, 3, 1, 4)
#         kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, self.inner_dim // self.heads).permute(2, 0, 3, 1, 4)
#         q = q[0]
#         k,v = kv[0], kv[1]
#         return q, k, v


# class SepConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, act_layer=nn.ReLU) -> None:
#         super(SepConv1d, self).__init__()
        
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride 
        
#         self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels)
#         self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
#         self.act_layer = act_layer() or nn.Identity()
        
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.act_layer(x)
#         x = self.pointwise(x)
#         return x
        
    
# class ConvProjection(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0., last_stage=False, bias=True) -> None:
        
#         super().__init__()
        
#         inner_dim = heads * dim_head
#         self.inner_dim = inner_dim
#         self.heads = heads 
#         pad = (kernel_size - q_stride)//2
        
#         self.to_q = SepConv1d(dim, inner_dim, kernel_size, q_stride, pad)
#         self.to_k = SepConv1d(dim, inner_dim, kernel_size, k_stride, pad)
#         self.to_v = SepConv1d(dim, inner_dim, kernel_size, v_stride, pad)
        
#     def forward(self, x, attn_kv=None):
#         b, n, c, h = *x.shape, self.heads
#         attn_kv = attn_kv or x
#         x = rearrange(x, 'b n c -> b c n')
#         attn_kv = rearrange(attn_kv, 'b n c -> b c n')
        
#         q = self.to_q(x)
#         # batch x inner_dim x length
#         q = rearrange(q, 'b (h d) n -> b h n d', h=h)
        
        
#         k = self.to_k(attn_kv)
#         # batch x inner_dim x length
#         k = rearrange(k, 'b (h d) n -> b h n d', h=h)
        
#         v = self.to_v(attn_kv)
#         # batch x inner_dim x length
#         v = rearrange(v, 'b (h d) n -> b h n d', h=h)
        
#         return q,k,v

# ################Attention#############
        
        
        
    
# ###############MLP####################
# class eca_layer_1d(nn.Module):
#     def __init__(self, channels, k_size=3) -> None:
#         super(eca_layer_1d, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.channels = channels
#         self.k_size = k_size
        
#     def forward(self, x):
#         y = self.avg_pool(x.transpose(-1, -2))
#         y = self.conv(y.transpose(-1, -2))
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)
    
        
        

# class LeFF(nn.Module):
#     def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False) -> None:
#         super().__init__()
#         self.dim = dim
#         self.hidden_dim = hidden_dim
#         self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

#         self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
#         self.dwconv = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1), act_layer())
#         self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))

#     def forward(self, x):
#         bs, hw, c = x.size()
        
#         x = self.linear1(x)
    
#         x = rearrange(x, 'b n c -> b c n')
#         x = self.dwconv(x)
#         x = rearrange(x, 'b c n -> b n c')

#         x = self.linear2(x)
        
#         x = self.eca(x)
#         return x

        
# class FastLeFF(nn.Module):
#     def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.) -> None:
#         super().__init__()
#         self.dim = dim
#         self.hidden_dim = hidden_dim
    
#         self.linear1 = nn.Sequential(nn.Linear(dim ,hidden_dim), act_layer())
#         self.dwconv = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, stride=1), act_layer())
#         self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        
        
#     def forward(self, x):
#         b, n, c = x.size()
#         x = self.linear1(x)
        
#         x = rearrange(x, 'b n c -> b c n')
#         x = self.dwconv(x)
#         x = rearrange(x, 'b c n -> b n c')
        
#         x = self.linear2(x)
#         return x

# class MLP(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
#         super().__init__()
#         hidden_features = hidden_features or in_features
#         out_features = out_features or in_features

#         self.in_features = in_features
#         self.hidden_features = hidden_features
#         self.out_features = out_features
        
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
        
#     def forward(self, x):
#         x = self.act(self.fc1(x))
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x 
# #################Abs Pos Encoder################
# class PositionalEncoding(nn.Module):
#     """位置编码"""
#     def __init__(self, num_hiddens, dropout=0., max_len=1000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         # 创建一个足够长的P
#         self.P = torch.zeros((1, max_len, num_hiddens))
#         X = torch.arange(max_len, dtype=torch.float32).reshape(
#             -1, 1) / torch.pow(10000, torch.arange(
#             0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
#         self.P[:, :, 0::2] = torch.sin(X)
#         self.P[:, :, 1::2] = torch.cos(X)
        
#     def forward(self, X):
#         X = X + self.P[:, :X.shape[1], :].to(X.device)
#         return self.dropout(X)
        
# ##################Transformer Block#############      
# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads, 
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., 
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff', 
#                  modulator=False, cross_modulation=False) -> None:
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads 
#         self.mlp_ratio = mlp_ratio
#         self.token_mlp = token_mlp
        
    
        
#         self.attn = MSAttention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
#                                 attn_drop=attn_drop, proj_drop=drop, 
#                                 token_projection=token_projection)
#         self.norm1 = norm_layer(dim)
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         if token_mlp in ['ffn', 'mlp']:
#             self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         elif token_mlp == 'leff':
#             self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         elif token_mlp == 'fastleff':
#             self.mlp = FastLeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         else:
#             raise Exception('FFN Error!')
        
        


#     def forward(self, x, mask=None):
#         B, L, C = x.shape
        
#         x = self.norm1(x)
#         msa_in = x
        
#         attn = self.attn(msa_in)
#         ffn_in = attn + x 
        
        
        
        
        
#         return 
        
def np_normlize(nparray):    
    return (nparray - np.min(nparray, axis=0)) /(np.max(nparray, axis=0) - np.min(nparray, axis=0))
         
def calcu_SNR(signal, noise_signal, eps=1e-5):
    N = signal.size(0)
    signal_mean = torch.mean(signal.contiguous().view(N, -1) ** 2, dim=1)
    noise_mean = torch.mean((noise_signal-signal).contiguous().view(N, -1) ** 2, dim=1) + torch.FloatTensor([eps for i in range(N)])
    return 10 * torch.log10(torch.div(signal_mean, noise_mean))
        
def find_model(model_path, model_name, epoch=None):
    model_dirs = glob.glob(model_path)
    idxs = []
    epochs = []
    epoch = epoch or -1
    for i in range(len(model_dirs)):
        model_dir = model_dirs[i].split('_')
        if model_name.lower() ==  model_dir[-3].lower():
            idxs.append(i)
            if epoch == int(model_dir[-1].split('.')[0]):
                return model_dirs[i]
            epochs.append(int(model_dir[-1].split('.')[0]))
        
        
    idx = epochs.index(max(epochs))
    return model_dirs[idxs[idx]]
    
def second_order_difference_loss(a):
    """smooth loss

    Args:
        a (tensor): reconstructed signal
        dim (int, optional): . Defaults to 0.

    Returns:
        _type_: _description_
    """
    a = rearrange(a, 'b c l -> l b c')
    b = a[0:-2] + a[2:] - 2 * a[1:-1]
    b = torch.pow(b, 2)
    b = rearrange(b, 'l b c -> b (c l)')
    b = torch.mean(b, dim=1)
    return b

if __name__ == "__main__":
    a = torch.randn(2, 2, 10)
    
    print(second_order_difference_loss(a).shape)