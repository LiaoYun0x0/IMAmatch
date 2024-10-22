"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
from torch.nn import Module, Dropout
import torch.nn as nn
from einops import rearrange
import math

FIX_BUG_FOR_SCANNET = False

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1
    
class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None,height=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()
    

class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None,height=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class FocusedLinearAttention(Module):
    def __init__(self, dim,eps=1e-6,focusing_factor=3):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, 1, dim)))
        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, groups=dim, padding=2)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None,height=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = (kernel_function(queries) + 1e-6) / scale
        k = (kernel_function(keys) + 1e-6) / scale
        
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** self.focusing_factor
        k = k ** self.focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        kv = torch.einsum("nshd,nshv->nhdv", k, values)  # (S,D)' @ S,V
        z = 1 / (torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + self.eps)
        o = torch.einsum("nlhd,nhdv,nlh->nlhv", q, kv, z) 
        
        if FIX_BUG_FOR_SCANNET:
            height = 60
        vf = rearrange(values, 'n (h w) heads d -> (n heads) d h w',h=height).contiguous()
        vf = self.dwc(vf)
        vf = rearrange(vf, '(n heads) d h w -> n (h w) heads d', n=o.size(0)).contiguous()
        o = o + vf
        
        return o.contiguous()
    
class WindowChannelAttention(nn.Module):
    def __init__(self,d_head=32,w=8):
        super().__init__()
        self.d_head = d_head
        self.w = w
        
    def forward(self,q,k,v_spatial,v_channel):
        m1 = self.window_attention(q,k,v_spatial)
        m2 = self.channel_attention(q,k,v_channel)
        return m1 + m2  
    
    def window_attention(self,q,k,v):
        '''
        q,k,v: [N,C,H,W]
        '''
        n,c,h,w = q.shape
        m = h // self.w
        n = w // self.w
        q,k,v = map(lambda x:rearrange(q, 'b (h d) (m w1) (n w2) -> (b m n) h (w1 w2) d',d=self.d_head,w1=self.w,w2=self.w), [q,k,v])
        attn = q @ k.transpose(-1,-2) / q.size(3) ** 0.5
        attn = torch.softmax(attn, dim=-1)
        x = attn @ v
        x = rearrange(x,'(b m n) h (w1 w2) d -> b (h d) (m w1) (n w2)',m=m,n=n,w1=self.w)
        return x
        
    def channel_attention(self,q,k,v,d_head=128):
        '''
        q,k,v: [N,C,H,W]
        '''
        n,c,h,w = q.shape
        q,k,v = map(lambda x:x.view(n,c,-1,d_head).transpose(1,2),[q,k,v])
        attention = q @ k.transpose(-1,-2) / q.size(3) ** 0.5
        attention = torch.softmax(attention,dim=-1)
        x = attention @ v
        x = x.transpose(1,2).reshape(n,c,h,w)
        return x

