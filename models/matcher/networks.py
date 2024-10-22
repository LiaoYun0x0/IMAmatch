import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_
import math

class ConvBNGelu(nn.Module):
    def __init__(self,c_in,c_out,k,s):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in,c_out,k,s,k//2,bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU()
        )
    def forward(self,x):
        return self.net(x)

class ImageLayerNorm(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self,x):
        return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
    
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class BiDimentionFFN(nn.Module):
    def __init__(self,dim_in,r=4):
        super().__init__()
        hidden_dim = dim_in // r
        self.global_branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.GELU()
        )
        self.global_branch2 = nn.Sequential(
            nn.Conv2d(hidden_dim, dim_in, 1),
            nn.Sigmoid()
        )
        self.local_branch1 = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.GELU()
        )
        self.local_branch2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, 1, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        gf = self.global_branch1(x)
        lf = self.local_branch1(x)
        lf = torch.cat([lf,torch.tile(gf, (1,1,lf.shape[2],lf.shape[3]))],dim=1)
        channel_atten = self.global_branch2(gf)
        spatial_attn = self.local_branch2(lf)
        attn = channel_atten * spatial_attn
        x = x * attn
        return x

class LinearFeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

class SE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(in_dim, hidden_dim,1, bias=False),
                                  nn.GELU(),
                                  nn.Conv2d(hidden_dim, in_dim,1, bias=False),
                                  nn.Sigmoid())
    def forward(self, x):
        return x * self.gate(x.mean((2,3), keepdim=True)) 
    
class MBConv(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size=3,stride_size=1,expand_rate = 4,se_rate = 0.25,dropout = 0.):
        super().__init__()
        hidden_dim = int(expand_rate * out_dim)
        self.bn = nn.BatchNorm2d(in_dim)
        self.expand_conv = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
                                         nn.BatchNorm2d(hidden_dim),
                                         nn.GELU())
        self.dw_conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride_size, kernel_size//2, groups=hidden_dim, bias=False),
                                     nn.BatchNorm2d(hidden_dim),
                                     nn.GELU())
        self.se = SE(hidden_dim,max(1,int(out_dim*se_rate)))
        self.out_conv = nn.Sequential(nn.Conv2d(hidden_dim, out_dim, 1, bias=False),
                                      nn.BatchNorm2d(out_dim))
        if stride_size > 1:
            self.proj = nn.Sequential(nn.MaxPool2d(kernel_size, stride_size, kernel_size//2),
                                      nn.Conv2d(in_dim, out_dim, 1, bias=False)) 
        else: 
            if in_dim == out_dim:
                self.proj = nn.Identity()
            else:
                self.proj = nn.Conv2d(in_dim, out_dim, 1,bias=False)
    
    def forward(self, x):
        out = self.bn(x)
        out = self.expand_conv(out)
        out = self.dw_conv(out)
        out = self.se(out)
        out = self.out_conv(out)
        return out + self.proj(x)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x