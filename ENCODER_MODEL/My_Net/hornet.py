# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import os
import sys
import torch.fft
import math
from .CBAM import CBAM
import traceback
from Models.ENCODER_MODEL.My_Net.lpf import Downsample

import torch.utils.checkpoint as checkpoint


def get_dwconv(dim, kernel, bias):
        return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

    # print('Using PyTorch large kernel dw conv impl')


class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s

        print('[gconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


class Block(nn.Module):
    r""" HorNet block
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Dence_block(nn.Module):
    def __init__(self,in_channels,channels):
        super(Dence_block, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(4):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=channels,out_channels=channels,stride=1,kernel_size=3,padding=1),
                    nn.ReLU()
                )
            )
        self.first = nn.Conv2d(in_channels=in_channels,out_channels=channels,stride=1,kernel_size=3,padding=1)
        self.last = nn.Conv2d(in_channels=channels,out_channels=channels,stride=1,kernel_size=3,padding=1)

    def forward(self,x):
        x = self.first(x)
        out = []
        out.append(x)
        for layer in self.blocks:
            x = layer(x)
            if len(out)>0:
                for i in out:
                    x = i + x
            out.append(x)
        return self.last(x)

class RRDB(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(RRDB, self).__init__()
        self.blocks = nn.ModuleList()
        self.beta = 0.9
        for i in range(3):
            self.blocks.append(Dence_block(channels=out_channel))
        self.start = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,stride=1,kernel_size=3,padding=1)
        self.last = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, stride=1, kernel_size=3,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, stride=1, kernel_size=3,padding=1)
        )
    def forward(self,x):
        x = self.start(x)
        out = x
        for layer in self.blocks:
            x = x + layer(x)*self.beta
        x = out+self.beta*x
        return self.last(x)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,3, stride,padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes,3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.inp = inplanes
        self.pla = planes

        if inplanes != planes:
            self.res_conv = nn.Conv2d(inplanes,planes,1,1,0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual if self.inp ==self.pla else self.res_conv(residual)
        out = self.relu(out)
        return out


class Cov_block(nn.Module):                 #定义卷积模块
    def __init__(self,in_chanels,med_chanels,out_chanels,padding = (1,1)):
        super(Cov_block,self).__init__()
        self.Cov1 = nn.Sequential(
            nn.Conv2d(in_chanels, med_chanels, kernel_size=(3,3), stride=1, padding=padding),
            nn.BatchNorm2d(med_chanels),
            nn.ReLU(),)
        self.Cov2 = nn.Sequential(
            nn.Conv2d(med_chanels, out_chanels, kernel_size=(3,3), stride=1, padding=padding),
            nn.BatchNorm2d(out_chanels),
            nn.ReLU(),
        )
    def forward(self,x):
        x = self.Cov1(x)
        x = self.Cov2(x)
        return x



class MSB(nn.Module):
    def __init__(self,in_channel,dim,out_channel):
        super(MSB, self).__init__()
        dim = out_channel//2
        self.pool = nn.AvgPool2d(kernel_size=2,stride=2)
        # self.pool = Downsample(channels = in_channel)
        # self.pool = nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=2,stride=2)
        # self.super_conv = nn.Sequential(
        #     nn.Conv2d(in_channel,out_channel,3,1,1),nn.BatchNorm2d(out_channel),nn.ReLU(),
        #     nn.Conv2d(out_channel,out_channel,3,2,1))
        # self.up = nn.ConvTranspose2d(in_channels=in_channel,out_channels=in_channel,stride=2,kernel_size=2)
        
        self.super_conv = nn.Sequential(nn.Conv2d(in_channel,out_channel,3,2,1))
       
        self.conv = BasicBlock(inplanes=in_channel,planes=out_channel)
        # self.conv = Cov_block(in_chanels=in_channel,med_chanels=nout_channel,out_chaels=out_channel)
    
        self.together_conv = CBAM(gate_channels=2*out_channel,no_spatial=True)
        
        self.res = BasicBlock(inplanes=out_channel*2,planes=out_channel)
        # self.res = Cov_block(in_chanels=out_channel*2,med_chanels=out_channel,out_chanels=out_channel) 

    def forward(self,x):
        super = F.interpolate(x, scale_factor=2, mode="nearest")
        # super = self.up(x)
        Low = self.pool(x)
        
        candy = super - F.interpolate(Low,scale_factor=4, mode="nearest")
        candy = self.super_conv(candy)
        x = self.conv(x)
        x = torch.cat((x,candy),dim=1)
        x = self.together_conv(x)
        x = self.res(x)
        return x


class HorNet(nn.Module):
    r""" HorNet
        A PyTorch impl of : `HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions`

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[2, 3, 6, 2], base_dim=64, drop_path_rate=0.4,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 gnconv=[
            'partial(gnconv, order=2, s=1/3)',
            'partial(gnconv, order=3, s=1/3)',
            'partial(gnconv, order=4, s=1/3, h=24, w=13, gflayer=GlobalLocalFilter)',
            'partial(gnconv, order=5, s=1/3, h=12, w=7, gflayer=GlobalLocalFilter)',
        ], block=Block, out_indices=[0, 1, 2,3],
                 pretrained=None,
                 use_checkpoint=False,
                 ):
        super().__init__()

        self.out_indices = out_indices
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint

        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        

        self.first_stage = MSB(in_channel=3,dim=16,out_channel=32)
        # self.first_stage = nn.Sequential(
        #     BasicBlock(inplanes=3, planes = 32),
        #     BasicBlock(inplanes=32, planes = 32)
        # )
        self.first_down = nn.Sequential(
                LayerNorm(32, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(32, 64, kernel_size=2, stride=2),
            )
        self.second_stage = MSB(in_channel=64,dim=32,out_channel=64)
        # self.second_stage = nn.Sequential(
        #     BasicBlock(inplanes=64, planes = 64),
        #     BasicBlock(inplanes=64, planes = 64)
        # )
        
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(32, dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(gnconv, list):
            gnconv = [gnconv, gnconv, gnconv, gnconv]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        if isinstance(gnconv[0], str):
            print('[GConvNet]: convert str gconv to func')
            gnconv = [eval(g) for g in gnconv]

        if isinstance(block, str):
            block = eval(block)

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value, gnconv=gnconv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        

    def init_weights(self):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pretrained = self.pretrained

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # if isinstance(pretrained, str):
        #     self.apply(_init_weights)
        #     logger = get_root_logger()
        #     load_checkpoint(self, pretrained, strict=False, logger=logger)
        # elif pretrained is None:
        #     raise NotImplementedError()
        #     self.apply(_init_weights)
        # else:
        #     raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        outs = []
        x = self.first_stage(x)
        outs.append(x)
        x = self.first_down(x)
        x = self.second_stage(x)
        outs.append(x)
        # print(x.shape)
        for i in range(1,4):
            x = self.downsample_layers[i](x)

            if self.use_checkpoint:
                x = checkpoint.checkpoint_sequential(self.stages[i], 2, x)
            else:
                x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return x_out,tuple(outs[:-1])

    def forward(self, x):
        x,feature = self.forward_features(x)
        return x,feature




class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Cov_block(nn.Module):                 #定义卷积模块
    def __init__(self,in_chanels,med_chanels,out_chanels,padding = (1,1)):
        super(Cov_block,self).__init__()
        self.Cov1 = nn.Sequential(
            nn.Conv2d(in_chanels, med_chanels, kernel_size=(3,3), stride=1, padding=padding),
            nn.BatchNorm2d(med_chanels),
            nn.ReLU(),)
        self.Cov2 = nn.Sequential(
            nn.Conv2d(med_chanels, out_chanels, kernel_size=(3,3), stride=1, padding=padding),
            nn.BatchNorm2d(out_chanels),
            nn.ReLU(),
        )
    def forward(self,x):
        x = self.Cov1(x)
        x = self.Cov2(x)
        return x





