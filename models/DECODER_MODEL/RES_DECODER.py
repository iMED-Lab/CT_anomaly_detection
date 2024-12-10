import torch
import torch.nn as nn
from torchvision.models import vit_b_16
import random
from typing import Type, Any, Callable, Union, List, Optional
from torch import nn, einsum
import sys
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Type, Any, Callable, Union, List, Optional

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class ExcludeCLS(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        cls_token, x = x[:, :1], x[:, 1:]
        x = self.fn(x, **kwargs)
        return torch.cat((cls_token, x), dim = 1)
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 2,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, inplanes, stride)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, inplanes, stride)
        self.bn2 = norm_layer(inplanes)
        self.conv3 = conv3x3(inplanes, inplanes, stride)
        self.bn3 = norm_layer(inplanes)
        self.downsample = downsample
        self.stride = stride
        self.conv1x1 = nn.Conv2d(in_channels=inplanes,out_channels=planes,kernel_size=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = self.relu(out)
        out = self.conv1x1(out)
        return out


# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         vit_pre = vit_b_16(pretrained=True)


# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(
#                 Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)))
                
            
#     def forward(self, x):
#         for attn in self.layers:
#             x = attn(x)
#         return x
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            # x = ff(x)
        return x
    
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels,
            last_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.ConvTranspose2d(in_channels=last_channels,out_channels=in_channels,kernel_size=2,stride=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(self, in_channels=[256, 128, 64], skip_channels=[256, 128, 64], out_channels=1, rqchannel=None,use_sig=True):
        super().__init__()
        out_channels = out_channels
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.rqchannel = rqchannel
        if self.rqchannel is not None:
            self.first = nn.Conv2d(in_channels=rqchannel, out_channels=512, kernel_size=1)

        self.blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            self.blocks.append(
                DecoderBlock(in_channels[i], in_channels[i], skip_channels[i],
                             in_channels[i]*2 if i==0 else in_channels[i-1])
            )
        if use_sig:
            self.last = nn.Sequential(
                nn.Conv2d(in_channels=in_channels[-1], out_channels=out_channels, kernel_size=1, stride=1),
                nn.Sigmoid()
            )
        else:
            self.last = nn.Conv2d(in_channels=in_channels[-1], out_channels=out_channels, kernel_size=1, stride=1)
        self.use_sig = use_sig

        # self.label_t = nn.Parameter(torch.ones(16,25,1,dtype=int),requires_grad=False)
        # self.label = nn.Parameter(torch.ones(16,231,1,dtype=int),requires_grad=False)
    

    def forward(self, x, features=None):
        b, c, h, w = x.size()
        # x_ = x.detach()
        # y_ = y.detach()
        # out_r1 = out_r.view(b,512,-1).permute(0, 2, 1).contiguous().detach()
        # out_r1 = self.transformer(out_r1)
        # out_r1 = out_r1.permute(0, 2, 1).contiguous()
        # out_r1 = out_r1.view(b,c,h,w)

        if features is not None:
            features = features[::-1]
        if self.rqchannel is not None:
            x = self.first(x)

            # out_r1 = self.first(out_r1)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None and self.skip_channels[i]!=0:
                skip = features[i] if i < len(features) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
    
            # out_r1 = decoder_block(out_r1, skip=skip)
        
        return self.last(x)
    
class DecoderCup_vit(nn.Module):
    def __init__(self, in_channels=[256, 128, 64], skip_channels=[256, 128, 64], out_channels=1, rqchannel=None,use_sig=True):
        super().__init__()
        out_channels = out_channels
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.rqchannel = rqchannel
        if self.rqchannel is not None:
            self.first = nn.Conv2d(in_channels=rqchannel, out_channels=512, kernel_size=1)

        self.blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            self.blocks.append(
                DecoderBlock(in_channels[i], in_channels[i], skip_channels[i],
                             in_channels[i]*2 if i==0 else in_channels[i-1])
            )
        if use_sig:
            self.last = nn.Sequential(
                nn.Conv2d(in_channels=in_channels[-1], out_channels=out_channels, kernel_size=1, stride=1),
                nn.Sigmoid()
            )
        else:
            self.last = nn.Conv2d(in_channels=in_channels[-1], out_channels=out_channels, kernel_size=1, stride=1)
        self.use_sig = use_sig

    def forward(self, x, features=None):
        if features is not None:
            features = features[::-1]
        if self.rqchannel is not None:
            x = self.first(x)

            # out_r1 = self.first(out_r1)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None and self.skip_channels[i]!=0:
                skip = features[i] if i < len(features) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)

            # out_r1 = decoder_block(out_r1, skip=skip)
        
        return self.last(x)
    
class decoder_main(nn.Module):
    def __init__(self, in_channels=[256, 128, 64], skip_channels=[256, 128, 64], out_channels=1, rqchannel=None,use_sig=True):
        super().__init__()
        self.decoder_ori = DecoderCup(in_channels=in_channels,skip_channels=skip_channels,out_channels=out_channels)
        self.decoder_vit = DecoderCup_vit(in_channels=in_channels,skip_channels=skip_channels,out_channels=out_channels)
        # self.transformer = vit_b_16(pretrained=True)
        # self.conv = nn.Conv2d(in_channels=512,out_channels=768,kernel_size=1)
        # self.transformer = vit_b_16(pretrained=False).encoder
        # self.mlp_head = nn.Sequential(nn.LayerNorm(512),nn.Linear(512,2048))
        self.transformer = Transformer(512, 12, 16, 32, 2048, 0.1)
        # self.cross_entropy = nn.CrossEntropyLoss()
        # self.index = nn.Parameter(torch.randperm(0,256),requires_grad=False)
        self.index_code = nn.Parameter(torch.randperm(2048),requires_grad=False)
        # self.empty = nn.Parameter(torch.zeros(shape=[8,25,512]),requires_grad=False)
        # self.index_total_t= nn.Parameter(torch.zeros(size=[4,25,512],dtype=int), requires_grad=False)
        # self.index_total_r = nn.Parameter(torch.zeros(size=[4,231,512]), requires_grad=False)
        # self.label_t = nn.Parameter(torch.ones(16,25,1,dtype=int),requires_grad=False)
        # self.label = nn.Parameter(torch.ones(16,231,1,dtype=int),requires_grad=False)
    
    def random_vq(self, input, refer):
        b,c,h,w = input.size()
        n,c = refer.size()
        input = input.permute(0, 2, 3, 1).contiguous()
        input = input.view(b,h*w,c)
        index_list = []
        index_list_r = []
        index_tensor_t = torch.zeros(size=[b,int(h*w*0.1),c],dtype=int).to(refer.device)
        index_tensor_r = torch.zeros(size=[b,int(h*w*0.9+1),c],dtype=int).to(refer.device)
        refer_zero = torch.zeros(size=[int(h*w*0.1),c],dtype=int).to(refer.device)
        # trans_input_t = torch.zeros(size=[b,int(h*w*0.1),c]).to(refer.device)
        # trans_input_r = torch.zeros(size=[b,int(h*w*0.9+1),c]).to(refer.device)
        for idx in range(b):
            # indices = list(range(h*w))
            # indices1 = list(range(n))
            random_indices = torch.randperm(h*w).to(refer.device)

            random_indices_code = torch.randperm(n).to(refer.device)
            # random_indices[0:25].unsqueeze(0)
            # index_list.append(random_indices[0:25].unsqueeze(0))
            # index_list_r.append(random_indices[25:].unsqueeze(0))
            # random_indices1 = random.sample(indices1, k=int(n))
            # print(random_indices)
            # print(random_indices1)
            # index_2 = []
            for index in range(int(h*w)):
                #  print(input[idx,random_indices[index],:].shape)
                 
                 if index<int(h*w*0.1):
                    index_tensor_t[idx][index][:] = random_indices[index]
                    # input[idx,random_indices[index],:] = refer[random_indices_code[index],:]
                    input[idx,random_indices[index],:] = refer_zero[index,:]
                    # trans_input_t[idx, index, :] = refer[random_indices_code[index],:]
                 else:
                    index_tensor_r[idx][index-int(h*w*0.1)][:] = random_indices[index]
                    # trans_input_r[idx, index-int(h*w*0.1), :] = input[idx,random_indices[index],:]
                         
        # index_tensor_t = self.index_total_t.clone()
        # index_tensor_r = self.index_total_r.clone()
        # index_tensor = torch.cat(index_list,dim=0)
        # index_tensor_r = torch.cat(index_list_r,dim=0)
        # input = input.permute(0, 2, 1).contiguous()
        # input = input.view(b,c,h,w)

        return input, index_tensor_t,index_tensor_r

    def forward(self, x, y, features=None):
        b, c, h, w = x.size()
        # x_ = x.detach()s
        # y_ = y.detach()
        
        trans_input, index_t_new, index_r_new = self.random_vq(x,y)

        # self.empty[:,:,0] = index_t
        # index_t_new = index_t.unsqueeze(-1).expand(b,25,512).clone()
        # for index in range(b):
        #     for index1 in range(int(h*w*0.1)):
        #         for i in range(512):
        #             index_t_new[index,index1,i] = i
        # print(index_t_new)
        # index_t_0[:,:,0] = 0
        # index_t_1 = index_t.unsqueeze(-1) + index_co
        # # index_t_1[:,:,0] = 1
        # index_t_new = torch.cat([index_t_0, index_t_1],dim=-1)
        # index_r_new = index_r.unsqueeze(-1) 
        # index_r_0[:,:,0] = 0
        # index_r_1 = index_r.unsqueeze(-1)
        # index_r_1[:,:,0] = 1
        # index_r_new = torch.cat([index_r_0, index_r_1],dim=-1)
        # index_t = index_t.unsqueeze(-1)[:,:,1] = 1
        x_q = trans_input.detach()
        # x_q = x_q.permute(0, 2, 1).contiguous()
        x_re = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        out_q = self.transformer(x_q)
        # out_q_t = torch.gather(out_q, dim=1, index=index_t_new)
        # # print(out_q_t-trans_input_t)
        # x_q_t = torch.gather(x_re, dim=1, index=index_t_new)
        # # out_q_t = self.mlp_head(out_q_t)
        # out_q_r = torch.gather(out_q, dim=1, index=index_r_new)
        # x_q_r = torch.gather(x_re, dim=1, index=index_r_new)
        # out_q_r = self.mlp_head(out_q_r)
        # print(cls.shape)
        trans_input = trans_input.permute(0, 2, 1).contiguous()
        trans_input = trans_input.view(b,c,h,w)
        out_q = out_q.permute(0, 2, 1).contiguous()
        out_q = out_q.view(b,c,h,w)
        
        x_r = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        out_r = self.transformer(x_r)
        out_r = out_r.permute(0, 2, 1).contiguous()
        out_r = out_r.view(b,c,h,w)

        x = self.decoder_ori(x)
        out_q_rec = self.decoder_vit(out_q)
        out_r = self.decoder_vit(out_r)
        out_trans = self.decoder_ori(trans_input)

        # out_r1 = out_r.view(b,512,-1).permute(0, 2, 1).contiguous().detach()
        # out_r1 = self.transformer(out_r1)
        # out_r1 = out_r1.permute(0, 2, 1).contiguous()
        # out_r1 = out_r1.view(b,c,h,w)s

        
            # out_r1 = decoder_block(out_r1, skip=skip)
        
        return x, out_q_rec, out_r, out_trans