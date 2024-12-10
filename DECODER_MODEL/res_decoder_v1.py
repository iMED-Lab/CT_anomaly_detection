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

    def cross_entropy_error(self, p, y):

        delta = 1e-7  # 添加一个微小值，防止负无穷(np.log(0))的情况出现
        p = torch.nn.Sigmoid()(p)
        y = torch.nn.Sigmoid()(y)

        return -torch.mean((y+ delta) * torch.log(p + delta) + (1 - (y+ delta)) * torch.log(1 - p + delta))
    
    def feature_select(self, x):

        px = torch.nn.Softmax(dim=1)(x)
        entropy = -torch.sum(px * torch.log2(px),dim=1)
        entropy_new,_ = torch.sort(entropy,dim=0)

        return entropy_new

    
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
        # x_ = x.detach()
        # y_ = y.detach()
        
       

        x = self.decoder_ori(x)  
  
        # ce_loss= self.cross_entropy_error(out_q_t, x_q_t.detach())

        # out_r1 = out_r.view(b,512,-1).permute(0, 2, 1).contiguous().detach()
        # out_r1 = self.transformer(out_r1)
        # out_r1 = out_r1.permute(0, 2, 1).contiguous()
        # out_r1 = out_r1.view(b,c,h,w)s
            # out_r1 = decoder_block(out_r1, skip=skip)
        
        return x
    
class decoder_main_encoder(nn.Module):
    def __init__(self, in_channels=[256, 128, 64], skip_channels=[256, 128, 64], out_channels=1, rqchannel=None,use_sig=True):
        super().__init__()
        self.decoder_ori = DecoderCup(in_channels=in_channels,skip_channels=skip_channels,out_channels=out_channels)
        self.decoder_vit = DecoderCup_vit(in_channels=in_channels,skip_channels=skip_channels,out_channels=out_channels)
        # self.transformer = vit_b_16(pretrained=True)
        # self.conv = nn.Conv2d(in_channels=512,out_channels=768,kernel_size=1)
        self.transformer = vit_b_16(pretrained=False).encoder
        # self.mlp_head = nn.Sequential(nn.LayerNorm(512),nn.Linear(512,2048))
        self.transformer = Transformer(512, 12, 16, 32, 2048, 0.1)
        self.mlp_head = nn.Sequential(nn.LayerNorm(512),nn.Linear(512,2))
        self.cross_entropy = nn.CrossEntropyLoss()
        # self.index = nn.Parameter(torch.randperm(0,256),requires_grad=False)
        self.index_code = nn.Parameter(torch.randperm(2048),requires_grad=False)
        self.pos_embedding = self.posemb_sincos_2d(h = 16, w = 16, dim = 512) 
    
     
    def feature_entropy(self, x):


        b,c,h,w = x.size()
        feature_tensor_reshaped = x.view(b,c, -1).permute(0, 2, 1).contiguous()
        prob = F.softmax(feature_tensor_reshaped, dim=1)
    
    # 计算特征信息熵
        entropy = -(prob * torch.log2(prob + 1e-10)).mean(dim=-1)  # 对每个样本计算信息熵，然后取平均
        # sorted_entropy, indices = torch.sort(entropy, descending=True)

        return entropy
    
    def var(self, x):
        b,c,h,w = x.size()
        feature_tensor_reshaped = x.view(b,c, -1).permute(0, 2, 1).contiguous()
        fea_var = torch.var(feature_tensor_reshaped,dim=-1)

        return fea_var
    
    def sort_fea(self, x):

        # entropy = nn.Sigmoid()(self.feature_entropy(x))
        fea_var = nn.Sigmoid()(self.var(x))
        total_fea = fea_var
        sorted_entropy, indices = torch.sort(total_fea, 1,descending=True)

        return sorted_entropy, indices
    
    def find_nearest_neighbors(self, features, codebook, k=20):
    # 将特征张量转换为2D张量（batch_size * num_features, feature_dim）
       
    
    # 计算特征张量与codebook张量之间的距离
        b,n,c = features.size()
        features_flat = features.reshape(b*n,c)
        distances = torch.cdist(features_flat, codebook)
    
    # 找到每个输入特征的最近邻索引
        _, nearest_neighbors_indices = torch.topk(distances, k=k, largest=False, dim=1)
    
        return nearest_neighbors_indices.view(features.size(0), features.size(1), k)
    
    def posemb_sincos_2d(self, h, w, dim, temperature: int = 10000, dtype = torch.float32):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature ** omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return pe.type(dtype)

    def random_vq(self, input, refer,sort_indices):
        b,c,h,w = input.size()
        b,n,c = refer.size()
        input = input.permute(0, 2, 3, 1).contiguous()
        input = input.view(b,h*w,c)
        index_list = []
        index_list_r = []
        index_tensor_t = torch.zeros(size=[b,int(h*w*0.1),2],dtype=int).to(refer.device)
        index_tensor_r = torch.zeros(size=[b,int(h*w*0.9+1),2],dtype=int).to(refer.device)
        # refer_zero = torch.zeros(size=[int(h*w*0.),c],dtype=int).to(refer.device)
        # trans_input_t = torch.zeros(size=[b,int(h*w*0.1),c]).to(refer.device)
        # trans_input_r = torch.zeros(size=[b,int(h*w*0.9+1),c]).to(refer.device)
        for idx in range(b):
            # indices = list(range(h*w))
            # indices1 = list(range(n))
            random_indices = torch.randperm(int(h*w)).to(refer.device)
            # random_indices = torch.arange(0,h*w*0.1).to(refer.device)
            random_indices_code = torch.randperm(n).to(refer.device)
            # random_indices[0:25].unsqueeze(0)
            # index_list.append(random_indices[0:25].unsqueeze(0))
            # index_list_r.append(random_indices[25:].unsqueeze(0))
            # random_indices1 = random.sample(indices1, k=int(n))
            # print(random_indices)
            # print(random_indices1)
            # index_2 = []
            a = random.uniform(0.25, 0.65)
            # for index in range(int(h*w)):
            #     #  print(input[idx,random_indices[index],:].shape)
            #      if int(h*w*a)>index:
            #          index_tensor_r[idx][index][:] = index
            #      if int(h*w*a)<=index<(int(h*w*a)+int(h*w*0.1)):
            #         index_tensor_t[idx][index-int(h*w*a)][:] = random_indices[index-int(h*w*a)]
            #         input[idx,random_indices[index-int(h*w*a)],:] = 1-refer[idx][random_indices_code[index-int(h*w*a)],:]
            #         # input[idx,random_indices[index],:] = refer_zero[index,:]
            #         # trans_input_t[idx, index, :] = refer[random_indices_code[index],:]
            #      if index>=(int(h*w*a)+int(h*w*0.1)):
            #         index_tensor_r[idx][index-(int(h*w*a)+int(h*w*0.1))][:] = index
                    # trans_input_r[idx, index-int(h*w*0.1), :] = input[idx,random_indices[index],:]
            for index in range(int(h*w)):
                #  print(input[idx,random_indices[index],:].shape)
                 
                 if index<int(h*w*0.1):
                    index_tensor_t[idx][index][:] = sort_indices[idx][index]
                    input[idx,sort_indices[idx][index],:] = refer[idx][random_indices_code[index],:]
                    # input[idx,random_indices[index],:] = refer_zero[index,:]
                    # trans_input_t[idx, index, :] = refer[random_indices_code[index],:]
                 else:
                    index_tensor_r[idx][index-int(h*w*0.1)][:] = sort_indices[idx][index]
                         
        # index_tensor_t = self.index_total_t.clone()
        # index_tensor_r = self.index_total_r.clone()
        # index_tensor = torch.cat(index_list,dim=0)
        # index_tensor_r = torch.cat(index_list_r,dim=0)
        # input = input.permute(0, 2, 1).contiguous()
        # input = input.view(b,c,h,w)

        return input, index_tensor_t, index_tensor_r

    def forward(self, encoder, x, y, features=None):
        print(encoder.size())
        b, c, h, w = x.size()
        sort_value, sort_indices = self.sort_fea(encoder)
        sort_fea = torch.index_select(encoder.view(b*h*w,-1), dim=0, index=sort_indices.view(-1))
        # sort_25_indices = sort_indices[:,0:25]

        sort_fea_25 = sort_fea.view(b,h*w,c)[:,0:25,:]
        
        att_indices = self.find_nearest_neighbors(sort_fea_25,y)
        att_fea = torch.index_select(y, 0, att_indices.view(-1))
        att_fea = att_fea.view(b,-1,512)

        ori_codes_t = torch.zeros(size=[b,int(h*w*0.1)],dtype=int).to(x.device)
        ori_codes_r = torch.ones(size=[b,int(h*w*0.9+1)],dtype=int).to(x.device)
        trans_input, index_t_new, index_r_new = self.random_vq(x,att_fea,sort_indices)

        x_q = trans_input.detach()
        # x_q = x_q.permute(0, 2, 1).contiguous()
        x_re = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        x_q += self.pos_embedding.to(x_q.device)
        out_q = self.transformer(x_q)
        out_q = self.mlp_head(out_q)
        out_q_t = torch.gather(out_q, dim=1, index=index_t_new)
        out_q_r = torch.gather(out_q, dim=1, index=index_r_new)
        ce_loss_t = self.cross_entropy(out_q_t.view(-1,2), ori_codes_t.view(-1).detach()) 
        ce_loss_r = self.cross_entropy(out_q_r.view(-1,2), ori_codes_r.view(-1).detach()) 
        ce_loss = ce_loss_t + ce_loss_r
        out_q = torch.nn.Softmax(dim=1)(out_q.view(b*h*w,-1))
        post_pro = out_q[:,0].view(b,1,h,w)
        post_pro[post_pro<0.5] = 0.0

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
        # out_q = out_q.permute(0, 2, 1).contiguous()
        # out_q = out_q.view(b,c,h,w)
        
        # x_r = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        # out_r = self.transformer(x_r)
        # out_r = out_r.permute(0, 2, 1).contiguous()
        # out_r = out_r.view(b,c,h,w)

        x = self.decoder_ori(x)
        # out_q_rec = self.decoder_vit(out_q)
        # out_r = self.decoder_vit(out_r)
        out_trans = self.decoder_ori(trans_input)
        out_trans_new = self.decoder_vit(trans_input)


        # out_r1 = out_r.view(b,512,-1).permute(0, 2, 1).contiguous().detach()
        # out_r1 = self.transformer(out_r1)
        # out_r1 = out_r1.permute(0, 2, 1).contiguous()
        # out_r1 = out_r1.view(b,c,h,w)s

        
            # out_r1 = decoder_block(out_r1, skip=skip)
        
        return x,post_pro, out_trans, out_trans_new, ce_loss

    
        
    def test(self, x, y,codes):
        b, c, h, w = x.size()
        # trans_input, index_t_new, index_r_new = self.random_vq(x,y)
        # x_ = x.detach()
        # y_ = y.detach()
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
        # x_q = trans_input.detach()
        # x_q = x_q.permute(0, 2, 1).contiguous()
        # x_q = trans_input.detach()
        # x_q = x_q.permute(0, 2, 1).contiguous()
        x_re = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        # out_q = self.transformer(x_q)
        x_re = self.transformer(x_re)
        x_pro = self.mlp_head(x_re)
        out_q = torch.nn.Softmax(dim=1)(x_pro.view(b*h*w,-1))
        post_pro = out_q[:,0].view(b,1,h,w)
        post_pro[post_pro<0.5] = 0.0
        # x = x.permute(0, 2, 1).contiguous()
        # x = x.view(b,c,h,w)=
        
        # out_q_t = torch.gather(out_q, dim=1, index=index_t_new)
        # # print(out_q_t-trans_input_t)
        # x_q_t = torch.gather(x_re, dim=1, index=index_t_new)
        # # out_q_t = self.mlp_head(out_q_t)
        # out_q_r = torch.gather(out_q, dim=1, index=index_r_new)
        # x_q_r = torch.gather(x_re, dim=1, index=index_r_new)
        # out_q_r = self.mlp_head(out_q_r)
        # print(cls.shape)
        # trans_input = trans_input.permute(0, 2, 1).contiguous()
        # trans_input = trans_input.view(b,c,h,w)
        # out_q = out_q.permute(0, 2, 1).contiguous()
        # out_q = out_q.view(b,c,h,w)
        
        # x_r = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        # out_r = self.transformer(x_r)
        # out_r = out_r.permute(0, 2, 1).contiguous()
        # out_r = out_r.view(b,c,h,w)

        # x = self.decoder_ori(x)
        x_rec = self.decoder_ori(x)
        x_rec_new = self.decoder_vit(x)

        # out_r = self.decoder_vit(out_r)
        # out_trans = self.decoder_ori(trans_input)


        # out_r1 = out_r.view(b,512,-1).permute(0, 2, 1).contiguous().detach()
        # out_r1 = self.transformer(out_r1)
        # out_r1 = out_r1.permute(0, 2, 1).contiguous()
        # out_r1 = out_r1.view(b,c,h,w)s

        
            # out_r1 = decoder_block(out_r1, skip=skip)
        
        return x_rec, x_rec_new, post_pro


class decoder_main_encoder_rce(nn.Module):
    def __init__(self, in_channels=[256, 128, 64], skip_channels=[256, 128, 64], out_channels=1, rqchannel=None,use_sig=True):
        super().__init__()
        self.decoder_ori = DecoderCup(in_channels=in_channels,skip_channels=skip_channels,out_channels=out_channels)
        self.decoder_vit = DecoderCup_vit(in_channels=in_channels,skip_channels=skip_channels,out_channels=out_channels)
        # self.transformer = vit_b_16(pretrained=True)
        # self.conv = nn.Conv2d(in_channels=512,out_channels=768,kernel_size=1)
        # self.transformer = vit_b_16(pretrained=False).encoder
        # self.mlp_head = nn.Sequential(nn.LayerNorm(512),nn.Linear(512,2048))
        self.transformer = Transformer(512, 12, 16, 32, 2048, 0.1)
        self.mlp_head = nn.Sequential(nn.LayerNorm(512),nn.Linear(512,2))
        self.cross_entropy = nn.CrossEntropyLoss()
        # self.index = nn.Parameter(torch.randperm(0,256),requires_grad=False)
        self.index_code = nn.Parameter(torch.randperm(2048),requires_grad=False)
        # self.pos_embedding = self.posemb_sincos_2d(h = 16, w = 16, dim = 512)
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, 512))

    # def posemb_sincos_2d(self, h, w, dim, temperature: int = 10000, dtype = torch.float32):
    #     y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    #     assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    #     omega = torch.arange(dim // 4) / (dim // 4 - 1)
    #     omega = 1.0 / (temperature ** omega)

    #     y = y.flatten()[:, None] * omega[None, :]
    #     x = x.flatten()[:, None] * omega[None, :]
    #     pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    #     print(pe.shape)
    #     return pe.type(dtype)
     
    def feature_entropy(self, x):

        b,c,h,w = x.size()
        feature_tensor_reshaped = x.view(b,c, -1).permute(0, 2, 1).contiguous()
        prob = F.softmax(feature_tensor_reshaped, dim=1)
    
    # 计算特征信息熵
        entropy = -(prob * torch.log2(prob + 1e-10)).mean(dim=-1)  # 对每个样本计算信息熵，然后取平均
        # sorted_entropy, indices = torch.sort(entropy, descending=True)

        return entropy
    
    def var(self, x):
        b,c,h,w = x.size()
        feature_tensor_reshaped = x.view(b, c, -1).permute(0, 2, 1).contiguous()
        fea_var = torch.var(feature_tensor_reshaped,dim=-1)
        fea_var = (fea_var - fea_var.min()) / (fea_var.max() - fea_var.min())

        return fea_var
    
    def H_norm(self,x):

        B, C, H, W = x.shape
        pca_dim = 8

# 展平每个空间位置的通道值，形成 [256, 512] 的张量 (256 = 16x16)
        features_flat = x.view(B, C, -1).permute(0, 2, 1).contiguous() # [B, 256, 512]
        features_mean = features_flat.mean(dim=1, keepdim=True)  # [B, 1, C]
        features_centered = features_flat - features_mean        # 去中心化 [B, H*W, C]
        cov_matrix = torch.matmul(features_centered.transpose(1, 2), features_centered) / (H * W)  # [B, C, C]

    # # Step 3: 计算特征值和特征向量
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)  # eigvals: [B, C], eigvecs: [B, C, C]

    # # Step 4: 选择前 pca_dim 个主成分
        topk_indices = torch.argsort(eigvals, dim=-1, descending=True)[:, :pca_dim]  # [B, pca_dim]
        batch_indices = torch.arange(B, device=x.device).view(-1, 1).expand(-1, pca_dim)  # [B, 16]
        topk_eigvecs = eigvecs[batch_indices, :, topk_indices].reshape([B, C, pca_dim])  # [B, C, pca_dim]
    #     print(topk_eigvecs.size())
        # random_matrix = torch.randn(B, C, pca_dim).to(x.device)

    # Step 5: 投影降维
        features_pca = torch.matmul(features_flat, topk_eigvecs)  # [B, H*W, pca_dim]

    # Step 6: 恢复到空间分布维度
        features_pca = features_pca.view(B, pca_dim, H*W)

        cov_matrices = torch.matmul(features_pca, features_pca.transpose(-1, -2)) / (H * W)  # [B, C, C]
        eigenvalues = torch.linalg.eigh(cov_matrices).eigenvalues  # [B, C]

    # 归一化特征值
        lambda_norm = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True)  # [B, C]

    # 计算原始熵 H_A
        alpha = 0.1
        reg = torch.sum(lambda_norm**2, dim=-1)
        H_A = -torch.sum(lambda_norm * torch.log(lambda_norm + 1e-9), dim=-1) + alpha * reg

# 6. 计算删除特征后的表示熵 \( H_A^{\text{reduced}} \)
        H_A_per_position = []
        for i in range(H * W):  # 遍历每个空间位置
    # 删除该位置所有通道上的特征，即删除特征图的某个位置的所有特征
            features_flat_reduced = features_pca
            # features_flat_reduced_mean,_ = features_flat_reduced.min(dim=-1,keepdim=True)
            features_flat_reduced = torch.cat((features_pca[:, :, :i], features_pca[:,  :, i + 1:]), dim=-1)
            # features_flat_reduced[:, :, i] = 0 # 删除第 i 个位置的所有通道特征（即令它为 0）

    # 计算去除该位置特征后的表示熵
            # features_flat_reduced_centered = features_flat_reduced - features_flat_reduced.mean(dim=1, keepdim=True)
            cov_matrices_reduced = torch.matmul(features_flat_reduced, features_flat_reduced.transpose(-1, -2)) / (H * W)
            eigenvalues_reduced = torch.linalg.eigh(cov_matrices_reduced).eigenvalues
            lambda_norm_reduced = eigenvalues_reduced / eigenvalues_reduced.sum(dim=-1, keepdim=True)
    
    # 计算删除该位置后表示熵
            alpha = 0.1
            reg = torch.sum(lambda_norm**2, dim=-1)
            H_A_reduced = -torch.sum(lambda_norm_reduced * torch.log(lambda_norm_reduced + 1e-9), dim=-1) + alpha * reg
            H_A_per_position.append(H_A_reduced)

# 堆叠去除每个通道后的熵，结果为 [B, C]
        H_A_per_position = torch.stack(H_A_per_position, dim=1).to(x.device)  # [B, C]

# 7. 计算每个位置的熵差分 \( H(z_e^i) \)
        H_diff = abs(H_A.unsqueeze(-1) - H_A_per_position)  # [B, C]

# 8. 归一化熵分数 \( H_{\text{norm}}(z_e^i) \)，最终得到 [B, C]
        H_norm = (H_diff - H_diff.min(dim=-1, keepdim=True).values) / (
            H_diff.max(dim=-1, keepdim=True).values - H_diff.min(dim=-1, keepdim=True).values + 1e-8)  # [B, 256, C]

# 8. 归一化熵分数 \( H_{\text{norm}}(z_e^i) \)，最终得到 [B, 256]
         
        # print(H_norm)

        return H_norm
    
    def sort_fea(self, x):

        H_norm = self.H_norm(x)
        fea_var = self.var(x)
        total_fea = 0.5*H_norm + fea_var
        sorted_entropy, indices = torch.sort(total_fea, 1,descending=True)

        return sorted_entropy, indices
    
    def find_nearest_neighbors(self, features, codebook, k=20):
    # 将特征张量转换为2D张量（batch_size * num_features, feature_dim）
       
    # 计算特征张量与codebook张量之间的距离
        b,n,c = features.size()
        features_flat = features.reshape(b*n,c)
        distances = torch.cdist(features_flat, codebook)
    
    # 找到每个输入特征的最近邻索引
        _, nearest_neighbors_indices = torch.topk(distances, k=k, largest=False, dim=1)
    
        return nearest_neighbors_indices.view(features.size(0), features.size(1), k)

    def random_vq(self, input, refer,sort_indices):
        b,c,h,w = input.size()
        b,n,c = refer.size()
        input = input.permute(0, 2, 3, 1).contiguous()
        input = input.view(b,h*w,c)
        index_list = []
        index_list_r = []
        index_tensor_t = torch.zeros(size=[b,int(h*w*0.1),2],dtype=int).to(refer.device)
        index_tensor_r = torch.zeros(size=[b,int(h*w*0.9+1),2],dtype=int).to(refer.device)
        # refer_zero = torch.zeros(size=[int(h*w*0.),c],dtype=int).to(refer.device)
        # trans_input_t = torch.zeros(size=[b,int(h*w*0.1),c]).to(refer.device)
        # trans_input_r = torch.zeros(size=[b,int(h*w*0.9+1),c]).to(refer.device)
        for idx in range(b):
            # indices = list(range(h*w))
            # indices1 = list(range(n))
            random_indices = torch.randperm(int(h*w)).to(refer.device)
            # random_indices = torch.arange(0,h*w*0.1).to(refer.device)
            random_indices_code = torch.randperm(n).to(refer.device)
            # random_indices[0:25].unsqueeze(0)
            # index_list.append(random_indices[0:25].unsqueeze(0))
            # index_list_r.append(random_indices[25:].unsqueeze(0))
            # random_indices1 = random.sample(indices1, k=int(n))
            # print(random_indices)
            # print(random_indices1)
            # index_2 = []
            a = random.uniform(0.25, 0.65)
            # for index in range(int(h*w)):
            #     #  print(input[idx,random_indices[index],:].shape)
            #      if int(h*w*a)>index:
            #          index_tensor_r[idx][index][:] = index
            #      if int(h*w*a)<=index<(int(h*w*a)+int(h*w*0.1)):
            #         index_tensor_t[idx][index-int(h*w*a)][:] = random_indices[index-int(h*w*a)]
            #         input[idx,random_indices[index-int(h*w*a)],:] = 1-refer[idx][random_indices_code[index-int(h*w*a)],:]
            #         # input[idx,random_indices[index],:] = refer_zero[index,:]
            #         # trans_input_t[idx, index, :] = refer[random_indices_code[index],:]
            #      if index>=(int(h*w*a)+int(h*w*0.1)):
            #         index_tensor_r[idx][index-(int(h*w*a)+int(h*w*0.1))][:] = index
                    # trans_input_r[idx, index-int(h*w*0.1), :] = input[idx,random_indices[index],:]
            for index in range(int(h*w)):
                #  print(input[idx,random_indices[index],:].shape)
                 
                 if index<int(h*w*0.1):
                    index_tensor_t[idx][index][:] = sort_indices[idx][index]
                    input[idx,sort_indices[idx][index],:] = refer[idx][random_indices_code[index],:]
                    # input[idx,random_indices[index],:] = refer_zero[index,:]
                    # trans_input_t[idx, index, :] = refer[random_indices_code[index],:]
                 else:
                    index_tensor_r[idx][index-int(h*w*0.1)][:] = sort_indices[idx][index]
                         
        # index_tensor_t = self.index_total_t.clone()
        # index_tensor_r = self.index_total_r.clone()
        # index_tensor = torch.cat(index_list,dim=0)
        # index_tensor_r = torch.cat(index_list_r,dim=0)
        # input = input.permute(0, 2, 1).contiguous()
        # input = input.view(b,c,h,w)

        return input, index_tensor_t, index_tensor_r
    
    def tensor_dis(self,img1,img2):
    

        # 计算残差图
        img1_f = nn.AvgPool2d(2,2)(img1)
        img2_f = nn.AvgPool2d(2,2)(img2)
        residuals = img1_f - img2_f

        window_size = 4
        stride = 4

        # 使用 unfold 函数对残差图进行切分
        unfolded_residuals = F.unfold(residuals, kernel_size=(window_size, window_size), stride=stride)

        # 将切分后的残差图沿着通道维度合并
        unfolded_residuals = unfolded_residuals.view(unfolded_residuals.size(0), unfolded_residuals.size(1), -1)

        # 计算每个窗口的方差
        variance_per_window = torch.var(unfolded_residuals, dim=1)
        mean_per_window = torch.mean(unfolded_residuals, dim=1)
        # variance_per_window = torch.pow(variance_per_window,0.2)
        variance_per_window = torch.nn.functional.normalize(variance_per_window, p=2, dim=1) 
        mean_per_window = torch.nn.functional.normalize(mean_per_window, p=2, dim=1)



        return variance_per_window + mean_per_window
    
    def forward(self, img, encoder, x, y, features=None):

        b, c, h, w = x.size()
        # print(encoder.size())
        sort_value, sort_indices = self.sort_fea(encoder)
        sort_fea = torch.index_select(encoder.view(b*h*w,-1), dim=0, index=sort_indices.view(-1))
        # sort_25_indices = sort_indices[:,0:25]

        sort_fea_25 = sort_fea.view(b,h*w,c)[:,0:25,:]
        
        att_indices = self.find_nearest_neighbors(sort_fea_25,y)
        att_fea = torch.index_select(y, 0, att_indices.view(-1))
        att_fea = att_fea.view(b,-1,512)

        # ori_codes_t = torch.zeros(size=[b,int(h*w*0.1)],dtype=int).to(x.device)
        # ori_codes_r = torch.ones(size=[b,int(h*w*0.9+1)],dtype=int).to(x.device)
        trans_input, index_t_new, index_r_new = self.random_vq(x,att_fea,sort_indices)

        x_q = trans_input.detach()
    
        # x_q = x_q.permute(0, 2, 1).contiguous()
        # x_re = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        x_q_p = x_q + self.pos_embedding
        out_q = self.transformer(x_q_p)
        out_q = self.mlp_head(out_q)
        out_q_t = torch.gather(out_q, dim=1, index=index_t_new)
        out_q_r = torch.gather(out_q, dim=1, index=index_r_new)
        
        out_q = torch.nn.Softmax(dim=1)(out_q.view(b*h*w,-1))
        post_pro = out_q[:,0].view(b,1,h,w)
        post_pro[post_pro<0.5] = 0.0

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
        # out_q = out_q.permute(0, 2, 1).contiguous()
        # out_q = out_q.view(b,c,h,w)
        
        # x_r = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        # out_r = self.transformer(x_r)
        # out_r = out_r.permute(0, 2, 1).contiguous()
        # out_r = out_r.view(b,c,h,w)

        x_out = self.decoder_ori(x)
        # out_q_rec = self.decoder_vit(out_q)
        # out_r = self.decoder_vit(out_r)
        out_trans = self.decoder_ori(trans_input)
        out_trans_new = self.decoder_vit(trans_input)
        weight_var = self.tensor_dis(out_trans_new,img)
        weight_var_t = torch.gather(weight_var, dim=1, index=index_t_new[:,:,0])
        
        # print(weight_var_t.size())
        # ce_loss_t = self.cross_entropy(out_q_t.view(-1,2), ori_codes_t.view(-1).detach()) 
        # ce_loss_t = 
        # ce_loss_r = self.cross_entropy(out_q_r.view(-1,2), ori_codes_r.view(-1).detach()) 
        # ce_loss = ce_loss_t + ce_loss_r


        # out_r1 = out_r.view(b,512,-1).permute(0, 2, 1).contiguous().detach()
        # out_r1 = self.transformer(out_r1)
        # out_r1 = out_r1.permute(0, 2, 1).contiguous()
        # out_r1 = out_r1.view(b,c,h,w)s
        
            # out_r1 = decoder_block(out_r1, skip=skip)
        
        return x_out, post_pro, out_trans, out_trans_new, weight_var_t, out_q_t, out_q_r

    
    def test(self, x, y,codes):
        b, c, h, w = x.size()
        # trans_input, index_t_new, index_r_new = self.random_vq(x,y)
        # x_ = x.detach()
        # y_ = y.detach()
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
        # x_q = trans_input.detach()
        # x_q = x_q.permute(0, 2, 1).contiguous()
        # x_q = trans_input.detach()
        # x_q = x_q.permute(0, 2, 1).contiguous()
        x_re = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        # out_q = self.transformer(x_q)
        x_re_p = x_re + self.pos_embedding

        x_re_p = self.transformer(x_re_p)
        x_pro = self.mlp_head(x_re_p)
        out_q = torch.nn.Softmax(dim=1)(x_pro.view(b*h*w,-1))
        post_pro = out_q[:,0].view(b,1,h,w)
        post_pro[post_pro<0.5] = 0.0
        # x = x.permute(0, 2, 1).contiguous()
        # x = x.view(b,c,h,w)=
        
        # out_q_t = torch.gather(out_q, dim=1, index=index_t_new)
        # # print(out_q_t-trans_input_t)
        # x_q_t = torch.gather(x_re, dim=1, index=index_t_new)
        # # out_q_t = self.mlp_head(out_q_t)
        # out_q_r = torch.gather(out_q, dim=1, index=index_r_new)
        # x_q_r = torch.gather(x_re, dim=1, index=index_r_new)
        # out_q_r = self.mlp_head(out_q_r)
        # print(cls.shape)
        # trans_input = trans_input.permute(0, 2, 1).contiguous()
        # trans_input = trans_input.view(b,c,h,w)
        # out_q = out_q.permute(0, 2, 1).contiguous()
        # out_q = out_q.view(b,c,h,w)
        
        # x_r = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        # out_r = self.transformer(x_r)
        # out_r = out_r.permute(0, 2, 1).contiguous()
        # out_r = out_r.view(b,c,h,w)

        # x = self.decoder_ori(x)
        x_rec = self.decoder_ori(x)
        x_rec_new = self.decoder_vit(x)

        # out_r = self.decoder_vit(out_r)
        # out_trans = self.decoder_ori(trans_input)


        # out_r1 = out_r.view(b,512,-1).permute(0, 2, 1).contiguous().detach()
        # out_r1 = self.transformer(out_r1)
        # out_r1 = out_r1.permute(0, 2, 1).contiguous()
        # out_r1 = out_r1.view(b,c,h,w)s

        
            # out_r1 = decoder_block(out_r1, skip=skip)
        
        return x_rec, x_rec_new, post_pro


class decoder_main_encoder_VQ_FC(nn.Module):
    def __init__(self, in_channels=[256, 128, 64], skip_channels=[256, 128, 64], out_channels=1, rqchannel=None,use_sig=True):
        super().__init__()
        self.decoder_ori = DecoderCup(in_channels=in_channels,skip_channels=skip_channels,out_channels=out_channels)
        self.decoder_vit = DecoderCup_vit(in_channels=in_channels,skip_channels=skip_channels,out_channels=out_channels)
        # self.transformer = vit_b_16(pretrained=True)
        # self.conv = nn.Conv2d(in_channels=512,out_channels=768,kernel_size=1)
        # self.transformer = vit_b_16(pretrained=False).encoder
        # self.mlp_head = nn.Sequential(nn.LayerNorm(512),nn.Linear(512,2048))
        self.transformer = Transformer(512, 12, 16, 32, 2048, 0.1)
        self.mlp_head = nn.Sequential(nn.LayerNorm(512),nn.Linear(512,2))
        self.cross_entropy = nn.CrossEntropyLoss()
        # self.index = nn.Parameter(torch.randperm(0,256),requires_grad=False)
        self.index_code = nn.Parameter(torch.randperm(2048),requires_grad=False)
        # self.pos_embedding = self.posemb_sincos_2d(h = 16, w = 16, dim = 512)
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, 512))

    # def posemb_sincos_2d(self, h, w, dim, temperature: int = 10000, dtype = torch.float32):
    #     y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    #     assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    #     omega = torch.arange(dim // 4) / (dim // 4 - 1)
    #     omega = 1.0 / (temperature ** omega)

    #     y = y.flatten()[:, None] * omega[None, :]
    #     x = x.flatten()[:, None] * omega[None, :]
    #     pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    #     print(pe.shape)
    #     return pe.type(dtype)
     
    def feature_entropy(self, x):

        b,c,h,w = x.size()
        feature_tensor_reshaped = x.view(b,c, -1).permute(0, 2, 1).contiguous()
        prob = F.softmax(feature_tensor_reshaped, dim=1)
    
    # 计算特征信息熵
        entropy = -(prob * torch.log2(prob + 1e-10)).mean(dim=-1)  # 对每个样本计算信息熵，然后取平均
        # sorted_entropy, indices = torch.sort(entropy, descending=True)

        return entropy
    
    def var(self, x):
        b,c,h,w = x.size()
        feature_tensor_reshaped = x.view(b,c, -1).permute(0, 2, 1).contiguous()
        fea_var = torch.var(feature_tensor_reshaped,dim=-1)

        return fea_var
    
    def sort_fea(self, x):

        b,c,h,w = x.size()
        indices = torch.zeros(size=[b,int(h*w)],dtype=int).to(x.device)
        for i in range(b):
            random_permutation = torch.randperm(h*w)
            indices[i,:] = random_permutation


        return indices


    def find_nearest_neighbors(self, features, codebook, k=20):
    # 将特征张量转换为2D张量（batch_size * num_features, feature_dim）
       
    # 计算特征张量与codebook张量之间的距离
        b,n,c = features.size()
        features_flat = features.reshape(b*n,c)
        distances = torch.cdist(features_flat, codebook)
    
    # 找到每个输入特征的最近邻索引
        _, nearest_neighbors_indices = torch.topk(distances, k=k, largest=False, dim=1)
    
        return nearest_neighbors_indices.view(features.size(0), features.size(1), k)

    def random_vq(self, input, refer,sort_indices):
        b,c,h,w = input.size()
        b,n,c = refer.size()
        input = input.permute(0, 2, 3, 1).contiguous()
        input = input.view(b,h*w,c)
        index_list = []
        index_list_r = []
        index_tensor_t = torch.zeros(size=[b,int(h*w*0.1),2],dtype=int).to(refer.device)
        index_tensor_r = torch.zeros(size=[b,int(h*w*0.9+1),2],dtype=int).to(refer.device)
        # refer_zero = torch.zeros(size=[int(h*w*0.),c],dtype=int).to(refer.device)
        # trans_input_t = torch.zeros(size=[b,int(h*w*0.1),c]).to(refer.device)
        # trans_input_r = torch.zeros(size=[b,int(h*w*0.9+1),c]).to(refer.device)
        for idx in range(b):
            # indices = list(range(h*w))
            # indices1 = list(range(n))
            random_indices = torch.randperm(int(h*w)).to(refer.device)
            # random_indices = torch.arange(0,h*w*0.1).to(refer.device)
            random_indices_code = torch.randperm(n).to(refer.device)
            # random_indices[0:25].unsqueeze(0)
            # index_list.append(random_indices[0:25].unsqueeze(0))
            # index_list_r.append(random_indices[25:].unsqueeze(0))
            # random_indices1 = random.sample(indices1, k=int(n))
            # print(random_indices)
            # print(random_indices1)
            # index_2 = []
            a = random.uniform(0.25, 0.65)
            # for index in range(int(h*w)):
            #     #  print(input[idx,random_indices[index],:].shape)
            #      if int(h*w*a)>index:
            #          index_tensor_r[idx][index][:] = index
            #      if int(h*w*a)<=index<(int(h*w*a)+int(h*w*0.1)):
            #         index_tensor_t[idx][index-int(h*w*a)][:] = random_indices[index-int(h*w*a)]
            #         input[idx,random_indices[index-int(h*w*a)],:] = 1-refer[idx][random_indices_code[index-int(h*w*a)],:]
            #         # input[idx,random_indices[index],:] = refer_zero[index,:]
            #         # trans_input_t[idx, index, :] = refer[random_indices_code[index],:]
            #      if index>=(int(h*w*a)+int(h*w*0.1)):
            #         index_tensor_r[idx][index-(int(h*w*a)+int(h*w*0.1))][:] = index
                    # trans_input_r[idx, index-int(h*w*0.1), :] = input[idx,random_indices[index],:]
            for index in range(int(h*w)):
                #  print(input[idx,random_indices[index],:].shape)
                 
                 if index<int(h*w*0.1):
                    index_tensor_t[idx][index][:] = sort_indices[idx][index]
                    input[idx,sort_indices[idx][index],:] = refer[idx][random_indices_code[index],:]
                    # input[idx,random_indices[index],:] = refer_zero[index,:]
                    # trans_input_t[idx, index, :] = refer[random_indices_code[index],:]
                 else:
                    index_tensor_r[idx][index-int(h*w*0.1)][:] = sort_indices[idx][index]
                         
        # index_tensor_t = self.index_total_t.clone()
        # index_tensor_r = self.index_total_r.clone()
        # index_tensor = torch.cat(index_list,dim=0)
        # index_tensor_r = torch.cat(index_list_r,dim=0)
        # input = input.permute(0, 2, 1).contiguous()
        # input = input.view(b,c,h,w)

        return input, index_tensor_t, index_tensor_r
    
    def tensor_dis(self,img1,img2):
    

        # 计算残差图
        img1_f = nn.AvgPool2d(2,2)(img1)
        img2_f = nn.AvgPool2d(2,2)(img2)
        residuals = img1_f - img2_f

        window_size = 4
        stride = 4

        # 使用 unfold 函数对残差图进行切分
        unfolded_residuals = F.unfold(residuals, kernel_size=(window_size, window_size), stride=stride)

        # 将切分后的残差图沿着通道维度合并
        unfolded_residuals = unfolded_residuals.view(unfolded_residuals.size(0), unfolded_residuals.size(1), -1)

        # 计算每个窗口的方差
        variance_per_window = torch.var(unfolded_residuals, dim=1)
        # variance_per_window = torch.pow(variance_per_window,0.2)
        variance_per_window = torch.nn.functional.normalize(variance_per_window, p=2, dim=1) 


        return variance_per_window
    
    def forward(self, img, encoder, x, y, features=None):
        

        b, c, h, w = x.size()
        sort_indices = self.sort_fea(encoder)
        sort_fea = torch.index_select(encoder.view(b*h*w,-1), dim=0, index=sort_indices.view(-1))
        # sort_25_indices = sort_indices[:,0:25]
        # x_1 = x.view(b,h*w,c)

        # random_permutation = torch.randperm(x_1.size(1))
        # # print(random_permutation)
        # sort_fea = x_1[:, random_permutation, :]

        sort_fea_25 = sort_fea.view(b,h*w,c)[:,0:25,:]
        
        att_indices = self.find_nearest_neighbors(sort_fea_25,y)
        att_fea = torch.index_select(y, 0, att_indices.view(-1))
        att_fea = att_fea.view(b,-1,512)

        # ori_codes_t = torch.zeros(size=[b,int(h*w*0.1)],dtype=int).to(x.device)
        # ori_codes_r = torch.ones(size=[b,int(h*w*0.9+1)],dtype=int).to(x.device)
        trans_input, index_t_new, index_r_new = self.random_vq(x,att_fea,sort_indices)

        x_q = trans_input.detach()
    
        # x_q = x_q.permute(0, 2, 1).contiguous()
        # x_re = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        x_q_p = x_q + self.pos_embedding
        out_q = self.transformer(x_q_p)
        out_q = self.mlp_head(out_q)
        out_q_t = torch.gather(out_q, dim=1, index=index_t_new)
        out_q_r = torch.gather(out_q, dim=1, index=index_r_new)
        
        out_q = torch.nn.Softmax(dim=1)(out_q.view(b*h*w,-1))
        post_pro = out_q[:,0].view(b,1,h,w)
        post_pro[post_pro<0.5] = 0.0

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
        # out_q = out_q.permute(0, 2, 1).contiguous()
        # out_q = out_q.view(b,c,h,w)
        
        # x_r = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        # out_r = self.transformer(x_r)
        # out_r = out_r.permute(0, 2, 1).contiguous()
        # out_r = out_r.view(b,c,h,w)

        x_out = self.decoder_ori(x)
        # out_q_rec = self.decoder_vit(out_q)
        # out_r = self.decoder_vit(out_r)
        out_trans = self.decoder_ori(trans_input)
        out_trans_new = self.decoder_vit(trans_input)
        weight_var = self.tensor_dis(out_trans_new,img)
        weight_var_t = torch.gather(weight_var, dim=1, index=index_t_new[:,:,0])
        # print(weight_var_t.size())
        # ce_loss_t = self.cross_entropy(out_q_t.view(-1,2), ori_codes_t.view(-1).detach()) 
        # ce_loss_t = 
        # ce_loss_r = self.cross_entropy(out_q_r.view(-1,2), ori_codes_r.view(-1).detach()) 
        # ce_loss = ce_loss_t + ce_loss_r


        


        # out_r1 = out_r.view(b,512,-1).permute(0, 2, 1).contiguous().detach()
        # out_r1 = self.transformer(out_r1)
        # out_r1 = out_r1.permute(0, 2, 1).contiguous()
        # out_r1 = out_r1.view(b,c,h,w)s

        
            # out_r1 = decoder_block(out_r1, skip=skip)
        
        return x_out, post_pro, out_trans, out_trans_new, weight_var_t, out_q_t, out_q_r

    
        
    def test(self, x, y,codes):
        b, c, h, w = x.size()
        # trans_input, index_t_new, index_r_new = self.random_vq(x,y)
        # x_ = x.detach()
        # y_ = y.detach()
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
        # x_q = trans_input.detach()
        # x_q = x_q.permute(0, 2, 1).contiguous()
        # x_q = trans_input.detach()
        # x_q = x_q.permute(0, 2, 1).contiguous()
        x_re = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        # out_q = self.transformer(x_q)
        x_re_p = x_re + self.pos_embedding

        x_re_p = self.transformer(x_re_p)
        x_pro = self.mlp_head(x_re_p)
        out_q = torch.nn.Softmax(dim=1)(x_pro.view(b*h*w,-1))
        post_pro = out_q[:,0].view(b,1,h,w)
        post_pro[post_pro<0.5] = 0.0
        # x = x.permute(0, 2, 1).contiguous()
        # x = x.view(b,c,h,w)=
        
        # out_q_t = torch.gather(out_q, dim=1, index=index_t_new)
        # # print(out_q_t-trans_input_t)
        # x_q_t = torch.gather(x_re, dim=1, index=index_t_new)
        # # out_q_t = self.mlp_head(out_q_t)
        # out_q_r = torch.gather(out_q, dim=1, index=index_r_new)
        # x_q_r = torch.gather(x_re, dim=1, index=index_r_new)
        # out_q_r = self.mlp_head(out_q_r)
        # print(cls.shape)
        # trans_input = trans_input.permute(0, 2, 1).contiguous()
        # trans_input = trans_input.view(b,c,h,w)
        # out_q = out_q.permute(0, 2, 1).contiguous()
        # out_q = out_q.view(b,c,h,w)
        
        # x_r = x.view(b,512,-1).permute(0, 2, 1).contiguous()
        # out_r = self.transformer(x_r)
        # out_r = out_r.permute(0, 2, 1).contiguous()
        # out_r = out_r.view(b,c,h,w)

        # x = self.decoder_ori(x)
        x_rec = self.decoder_ori(x)
        x_rec_new = self.decoder_vit(x)

        # out_r = self.decoder_vit(out_r)
        # out_trans = self.decoder_ori(trans_input)


        # out_r1 = out_r.view(b,512,-1).permute(0, 2, 1).contiguous().detach()
        # out_r1 = self.transformer(out_r1)
        # out_r1 = out_r1.permute(0, 2, 1).contiguous()
        # out_r1 = out_r1.view(b,c,h,w)s

        
            # out_r1 = decoder_block(out_r1, skip=skip)
        
        return x_rec, x_rec_new, post_pro