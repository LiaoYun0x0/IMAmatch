import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
from kornia.utils import create_meshgrid
from .attention import LinearAttention,FullAttention,FocusedLinearAttention
from .networks import ConvBNGelu


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256),norm=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if norm:
            y_position = y_position / max_shape[0]
            x_position = x_position / max_shape[1]
        div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]
    
    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    
class MB(nn.Module):
    def __init__(self,dim_in,dim_out,mlp_ratio=4,prenorm=False,afternorm=True,k=3,stride=1):
        super().__init__()
        dim_mid = int(dim_in * mlp_ratio)
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim_in) if prenorm else nn.Identity(),
            nn.Conv2d(dim_in, dim_mid, 1,bias=False),
            nn.BatchNorm2d(dim_mid),
            nn.GELU(),
            nn.Conv2d(dim_mid, dim_mid, k,stride,k//2,groups=dim_mid,bias=False),
            nn.BatchNorm2d(dim_mid),
            nn.GELU(),
            nn.Conv2d(dim_mid, dim_out, 1,bias=False),
            nn.BatchNorm2d(dim_out) if afternorm else nn.Identity()
        )
    def forward(self,x):
        x = self.net(x)
        return x

class ResidualMB(nn.Module):
    def __init__(self,dim_in,dim_out,mlp_ratio=4,prenorm=False,afternorm=True,k=3,stride=1,dropout=0.):
        super().__init__()
        dim_mid = int(dim_in * mlp_ratio)
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim_in) if prenorm else nn.Identity(),
            nn.Conv2d(dim_in, dim_mid, 1,bias=False),
            nn.BatchNorm2d(dim_mid),
            nn.GELU(),
            nn.Conv2d(dim_mid, dim_mid, k,stride,1,groups=dim_mid,bias=False),
            nn.BatchNorm2d(dim_mid),
            nn.GELU(),
            nn.Conv2d(dim_mid, dim_out, 1,bias=False),
            nn.BatchNorm2d(dim_out) if afternorm else nn.Identity()
        )
        self.main = nn.Sequential(
            nn.MaxPool2d(k, stride, k//2) if stride > 1 else nn.Identity(),
            nn.Conv2d(dim_in, dim_out, 1, bias=False) if dim_in != dim_out else nn.Identity()
        )
        self.dropout = DropPath(dropout)
            
    def forward(self,x):
        return self.main(x) + self.dropout(self.net(x))
        
    
class ConvBlock(nn.Module):
    def __init__(self,dim,dropout=0.,mlp_ratio=4):
        super().__init__()
        self.conv = MB(dim,dim,mlp_ratio,False,True)
        self.mlp = MB(dim,dim,mlp_ratio,False,True)
        self.dropout = DropPath(dropout)
    def forward(self,x):
        x = x + self.dropout(self.conv(x))
        x = x + self.dropout(self.mlp(x))
        return x

class ImageAttentionLayer(nn.Module):
    def __init__(self,d_model,d_head,dropout=0.0,attention='linear',mlp_ratio=4):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.nhead = self.d_model // self.d_head
        
        self.pre_normact = nn.Sequential(
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.q_proj = nn.Conv2d(d_model, d_model, 1,1,bias=False)   
        self.k_proj = nn.Conv2d(d_model, d_model, 1,1,bias=False)  
        self.v_proj = nn.Conv2d(d_model, d_model, 1,1,bias=False)   

        if attention == 'linear':
            self.attention = LinearAttention()
        elif attention == 'full':
            self.attention = FullAttention()
        elif attention == 'focuse':
            self.attention = FocusedLinearAttention(dim=d_head)
        else:
            raise NotImplementedError()
        
        self.merge = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
        )
        self.mlp = MB(d_model,d_model,mlp_ratio,False,True)
        self.dropout = DropPath(dropout)
        
    def forward(self, x0,x1=None,q_mask=None,kv_mask=None):
        '''
        x0,x1: [N, C, H, W]
        q_mask,kv_mask: [N, (H W)]
        '''
        b,d,h,w = x0.shape
        if x1 is None:
            x1 = x0
        _x0,_x1 = self.pre_normact(x0),self.pre_normact(x1)
        q = self.q_proj(_x0)
        k = self.k_proj(_x1)
        v = self.v_proj(_x1)
        
        q = rearrange(q, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
        k = rearrange(k, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
        v = rearrange(v, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
        message = self.attention(q, k, v, q_mask=q_mask, kv_mask=kv_mask,height=h)  # [N, L, (H, D)]
        message = rearrange(message,'b (h w) heads d -> b (heads d) h w',h=h)
        
        x = x0 + self.dropout(self.merge(message.contiguous()))
        x = x + self.dropout(self.mlp(x))
        return x

class ImageAttentionBlock(nn.Module):
    def __init__(self,d_model,d_head,dropout=0.0,attention='linear',mlp_ratio=4):
        super().__init__()
        self.self_attn = ImageAttentionLayer(d_model, d_head,dropout,attention,mlp_ratio)
        self.cross_attn = ImageAttentionLayer(d_model, d_head,dropout,attention,mlp_ratio)
    def forward(self,x):
        x = self.self_attn(x)
        x0,x1 = torch.chunk(x, 2,dim=0)
        x0,x1 = self.cross_attn(x0,x1),self.cross_attn(x1,x0)
        x = torch.concat([x0,x1],dim=0)
        return x
    
class AttentionLayer(nn.Module):
    def __init__(self,d_model,d_head,dropout=0.0,attention='linear',mlp_ratio=4):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.nhead = self.d_model // self.d_head
        
        self.pre_norm = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        if attention == 'linear':
            self.attention = LinearAttention()
        elif attention == 'full':
            self.attention = FullAttention()
        elif attention == 'focuse':
            self.attention = FocusedLinearAttention(dim=d_head)
        else:
            raise NotImplementedError()
        
        self.merge = nn.Sequential(
            nn.Linear(d_model, d_model),
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*mlp_ratio,d_model),
            nn.Dropout(dropout)
        )
        self.dropout = DropPath(dropout)
        
    def forward(self, x0,x1=None,q_mask=None,kv_mask=None):
        '''
        x0,x1: [n, l, d]
        q_mask,kv_mask: [n, l]
        '''
        if x1 is None:
            x1 = x0
        _x0,_x1 = self.pre_norm(x0), self.pre_norm(x1)
        q = self.q_proj(_x0)
        k = self.k_proj(_x1)
        v = self.v_proj(_x1)
        
        q = rearrange(q, ' n l (h d) -> n l h d', h=self.nhead)
        k = rearrange(k, ' n s (h d) -> n s h d', h=self.nhead)
        v = rearrange(v, ' n s (h d) -> n s h d', h=self.nhead)
        message = self.attention(q, k, v, q_mask=None, kv_mask=None)
        message = rearrange(message,'n l h d -> n l (h d)')
        
        x = x0 + self.dropout(self.merge(message))
        x = x + self.dropout(self.mlp(x))
        return x
    

class AttentionBlock(nn.Module):
    def __init__(self,d_model,d_head,dropout=0.0,attention='linear',mlp_ratio=4):
        super().__init__()
        self.self_attn = AttentionLayer(d_model, d_head,dropout,attention,mlp_ratio)
        self.cross_attn = AttentionLayer(d_model, d_head,dropout,attention,mlp_ratio)
        
    def forward(self, x0,x1):
        x0, x1 = self.self_attn(x0,x0), self.self_attn(x1,x1)
        x0, x1 = self.cross_attn(x0, x1), self.cross_attn(x1,x0)
        return x0, x1
    

    

class MultiScaleAttentionLayer(nn.Module):
    def __init__(self,d_model,d_head,dropout=0.0,attention='linear',mlp_ratio=4,height=None):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.nhead = self.d_model // self.d_head
        self.register_parameter('scale_weight', torch.nn.Parameter(torch.tensor([1,1,1],dtype=torch.float32)))
        
        self.pre_normact = nn.Sequential(
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.q1_proj = nn.Conv2d(d_model, d_model, 1,1,bias=False)   
        self.k1_proj = nn.Conv2d(d_model, d_model, 1,1,bias=False)  
        self.v1_proj = nn.Conv2d(d_model, d_model, 1,1,bias=False)   
          
        self.q2_proj = nn.Conv2d(d_model, d_model, 2,2,bias=False) 
        self.k2_proj = nn.Conv2d(d_model, d_model, 2,2,bias=False)  
        self.v2_proj = nn.Conv2d(d_model, d_model, 2,2,bias=False)   
         
        self.q3_proj = nn.Conv2d(d_model, d_model, 4,4,bias=False) 
        self.k3_proj = nn.Conv2d(d_model, d_model, 4,4,bias=False)  
        self.v3_proj = nn.Conv2d(d_model, d_model, 4,4,bias=False)   

        if attention == 'linear':
            self.attention1 = LinearAttention()
            self.attention2 = LinearAttention()
            self.attention3 = LinearAttention()
        elif attention == 'full':
            self.attention1 = FullAttention()
            self.attention2 = FullAttention()
            self.attention3 = FullAttention()
        elif attention == 'focuse':
            self.attention1 = FocusedLinearAttention(dim=d_head)
            self.attention2 = FocusedLinearAttention(dim=d_head)
            self.attention3 = FocusedLinearAttention(dim=d_head)
        else:
            raise NotImplementedError()
        
        self.name = attention
        self.merge = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
        )
        self.mlp = MB(d_model,d_model,mlp_ratio,False,True)
        self.dropout = DropPath(dropout)
        
    def forward(self, x0,x1=None,q_mask=None,kv_mask=None):
        '''
        x0,x1: [N, C, H, W]
        q_mask,kv_mask: [N, (H W)]
        '''
        b,d,h,w = x0.shape
        if x1 is None:
            x1 = x0
        _x0,_x1 = self.pre_normact(x0),self.pre_normact(x1)
        q1 = self.q1_proj(_x0)
        k1 = self.k1_proj(_x1)
        v1 = self.v1_proj(_x1)
        q2 = self.q2_proj(_x0)
        k2 = self.k2_proj(_x1)
        v2 = self.v2_proj(_x1)
        q3 = self.q3_proj(_x0)
        k3 = self.k3_proj(_x1)
        v3 = self.v3_proj(_x1)
        
        q1 = rearrange(q1, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
        k1 = rearrange(k1, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
        v1 = rearrange(v1, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
        
        q2 = rearrange(q2, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
        k2 = rearrange(k2, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
        v2 = rearrange(v2, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
        
        q3 = rearrange(q3, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
        k3 = rearrange(k3, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
        v3 = rearrange(v3, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
        
        if self.name == 'focuse':
            o1 = self.attention1(q1,k1,v1,height=h)
            o2 = self.attention2(q2,k2,v2,height=h//2)
            o3 = self.attention3(q3,k3,v3,height=h//4)
        else:
            o1 = self.attention1(q1,k1,v1)
            o2 = self.attention2(q2,k2,v2)
            o3 = self.attention3(q3,k3,v3)
        
        o1 = rearrange(o1, 'b (h w) heads d -> b (heads d) h w', h=h)
        o2 = rearrange(o2, 'b (h w) heads d -> b (heads d) h w', h=h//2)
        o3 = rearrange(o3, 'b (h w) heads d -> b (heads d) h w', h=h//4)
        o2 = F.interpolate(o2,scale_factor=2,mode='bilinear',align_corners=True)
        o3 = F.interpolate(o3,scale_factor=4,mode='bilinear',align_corners=True)
        sw = torch.softmax(self.scale_weight, dim=0)
        message = sw[0]*o1 + sw[1]*o2 + sw[2]*o3
        
        x = x0 + self.dropout(self.merge(message.contiguous()))
        x = x + self.dropout(self.mlp(x))
        return x

class MultiScaleAttentionBlock(nn.Module):
    def __init__(self,d_model,d_head,dropout=0.0,attention='linear',mlp_ratio=4):
        super().__init__()
        self.self_attn = MultiScaleAttentionLayer(d_model, d_head,dropout,attention,mlp_ratio)
        self.cross_attn = MultiScaleAttentionLayer(d_model, d_head,dropout,attention,mlp_ratio)
    def forward(self,x):
        x = self.self_attn(x)
        x0,x1 = torch.chunk(x, 2,dim=0)
        x0,x1 = self.cross_attn(x0,x1),self.cross_attn(x1,x0)
        x = torch.concat([x0,x1],dim=0)
        return x
    

class PosEmb(nn.Module):
    def __init__(self,d_model=128,d_head=32,dropout=0.0,attention='linear',mlp_ratio=4,temperature=0.1,k=100,norm=False):
        super().__init__()
        self.temperature=temperature
        self.k = k
        self.embedding_linear = torch.nn.Linear(k*2, d_model)
        self.norm = norm

    def forward(self,x):
        x0,x1 = torch.chunk(x, chunks=2, dim=0)
        n,c,h,w = x0.shape
        f0 = rearrange(x0, 'n c h w -> n (h w) c')
        f1 = rearrange(x1, 'n c h w -> n (h w) c')
        f0_norm = f0 / f0.shape[-1] ** 0.5
        f1_norm = f1 / f1.shape[-1] ** 0.5
        sm = torch.einsum('nld,nsd->nls',f0_norm,f1_norm) / self.temperature
        cm = (torch.softmax(sm, dim=1) * torch.softmax(sm, dim=2)).view(n,-1)
        _,topi = torch.topk(cm,k=self.k,dim=-1)
        qindex = torch.div(topi,h*w,rounding_mode='trunc')
        rindex = topi % (h*w)
        coords0 = torch.stack([qindex % w, torch.div(qindex, w,rounding_mode='trunc')],dim=-1) / w
        coords1 = torch.stack([rindex % w, torch.div(rindex, w,rounding_mode='trunc')],dim=-1) / w
        grid = create_meshgrid(h, w, False,device=x0.device).reshape(1,-1,2) / w
        offsets0 = grid[:,:,None,:] - coords0[:,None,:,:]
        offsets1 = grid[:,:,None,:] - coords1[:,None,:,:]
        if self.norm:
            scale0 = torch.sqrt(torch.pow(coords0[:,:,None,:] - coords0[:,None,:,:],2).sum(dim=-1,keepdims=True)).mean(dim=(1,2,3),keepdims=True)
            scale1 = torch.sqrt(torch.pow(coords1[:,:,None,:] - coords1[:,None,:,:],2).sum(dim=-1,keepdims=True)).mean(dim=(1,2,3),keepdims=True)
            offsets0 = offsets0 / (scale0 / scale1)
        embedding0 = self.embedding_linear(offsets0.flatten(2)) 
        embedding1 = self.embedding_linear(offsets1.flatten(2))
        f0 = f0 + embedding0
        f1 = f1 + embedding1
        f = rearrange(torch.cat([f0,f1],dim=0),'n (h w) d -> n d h w',h=h)
        return f  

class MBFormer_248_standardPE(nn.Module):
    def __init__(
        self,
        *,
        dim_conv_stem = 64,
        dims=[128,192,256],
        depths=[2,2,2],
        d_spatial = 32,
        d_channel = 128,
        mbconv_expansion_rate = [1,1,2,3],
        dropout = 0.1,
        in_chans = 1,
        attn_depth = 4,
        attn_name = 'ImageAttentionBlock',
        w=None,
        k=None,
        attention='linear',
        img_size=(480,640)
    ):
        super().__init__()

        self.conv_stem = nn.Sequential(
            ConvBNGelu(in_chans, dim_conv_stem, 3, 2),
            ConvBNGelu(dim_conv_stem, dim_conv_stem, 3, 1),
        )
        self.conv_stem.requires_grad_()

        self.num_stages = len(dims)

        self.d0 = nn.Sequential(
                ConvBNGelu(dim_conv_stem, dims[0], 3, 1),
                ConvBNGelu(dims[0], dims[0], 3, 1),
            ) if depths[0] == 0 else \
            nn.Sequential(
                *([ResidualMB(dim_conv_stem, dims[0],stride=1,mlp_ratio=1)] + \
                [ConvBlock(dims[0],mlp_ratio=mbconv_expansion_rate[0]) for _ in range(depths[0])])
            )
            
        self.d1 = nn.Sequential(
            *([ResidualMB(dims[0], dims[1],stride=2,mlp_ratio=1)] + \
            [ConvBlock(dims[1],mlp_ratio=mbconv_expansion_rate[1]) for _ in range(depths[1])])
        )
        self.d2 = nn.Sequential(
            *([ResidualMB(dims[1], dims[2],stride=2,mlp_ratio=1)] + \
            [ConvBlock(dims[2],mlp_ratio=mbconv_expansion_rate[2]) for _ in range(depths[2])])
        )
        
        self.u0 = nn.ModuleList([
            nn.Conv2d(dims[0], dims[1], 1,bias=False),
            nn.Sequential(
                ConvBNGelu(dims[1], dims[1], 3, 1),
                nn.Conv2d(dims[1], dims[0], 3,1,1,bias=False)
            )
        ])
        self.u1 = nn.ModuleList([
            nn.Conv2d(dims[1], dims[2], 1,bias=False),
            nn.Sequential(
                ConvBNGelu(dims[2], dims[2], 3, 1),
                nn.Conv2d(dims[2], dims[1], 3,1,1,bias=False)
            )
        ])
        
        self.attn = nn.Sequential()
        for i in range(attn_depth):    
            atten_params = [dims[-1], d_spatial, dropout,attention,mbconv_expansion_rate[3]]
            self.attn.append(eval(attn_name)(*atten_params))
            
        # self.pos_emb = PositionEncodingSine(dims[-1],max_shape=(img_size[0]//8,img_size[1]//8))
    
    
    def forward(self, x):
        outputs = []
        x = self.conv_stem(x)
        x0 = self.d0(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        for i,attn in enumerate(self.attn):
            x2 = attn(x2)
        outputs.append(x2)
        x2_up = F.interpolate(x2, scale_factor=2., mode='bilinear', align_corners=True)
        x1 = self.u1[1](self.u1[0](x1) + x2_up)
        outputs.append(x1)
        x1_up = F.interpolate(x1, scale_factor=2., mode='bilinear', align_corners=True)
        x0 = self.u0[1](self.u0[0](x0) + x1_up)
        outputs.append(x0)
        return list(reversed(outputs))
        
        
    