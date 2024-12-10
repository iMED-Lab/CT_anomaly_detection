# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from inception import Inception3
# #########--------- Components ---------#########
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        
        return x



class conv_block1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2,inplace=True)
        )
    
    def forward(self, x):
        x = self.up(x)
        
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2,inplace=True)
        )
    
    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        
        return x * psi

class res_conv_block(nn.Module):
    def __init__(self, stride,ch_in, ch_out):
        super(res_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),

        )
        self.downsample = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
        )
        self.relu = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)

        return self.relu(out + residual)

class SKConv(nn.Module):
    def __init__(self, features, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            if i == 0:
                self.convs.append(nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=1, groups=features),
                    nn.BatchNorm2d(features),
                    nn.ReLU()))
            if i == 1:
                self.convs.append(nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=1, groups=features),
                    nn.BatchNorm2d(features),
                    nn.ReLU()))
            if i == 2:
                self.convs.append(nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=1,groups=features),
                    nn.BatchNorm2d(features),
                    nn.ReLU()))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1,x2,x3):
        x = [x1,x2,x3]
        for i, conv in enumerate(self.convs):
            fea = conv(x[i]).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            if i == 1:
                feas = torch.cat([feas, fea], dim=1)
            if i == 2:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            if i == 1:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
            if i == 2:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v
# #########--------- Networks ---------#########
class U_Net(nn.Module):
    def __init__(self, img_ch=3):
        super(U_Net, self).__init__()
        # self.Maxpool = nn.MaxPool2d(2)
        
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=256)
        # self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.Up5 = up_conv(ch_in=1024, ch_out=512)
        #
        # self.Up4 = up_conv(ch_in=1024,ch_out=256)
        #
        # self.Up3 = up_conv(ch_in=512, ch_out=128)
        #
        # self.Up2 = up_conv(ch_in=256, ch_out=64)
        #
        # self.Up1 = up_conv(ch_in=128, ch_out=1)
    
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Conv2(x1)
        
        x3 = self.Conv3(x2)
      
        x4 = self.Conv4(x3)
        

        
        return x4



class Encoder(nn.Module):
    def __init__(self, img_ch=3):
        super(Encoder, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # encoding path
        x0 = torch.cat([x,x,x],1)
        x0 = self.firstconv(x0)
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)
        x1 = self.firstmaxpool(x0)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        #x5 = self.encoder4(x4)

        # decoding + concat path

        # out = nn.Sigmoid()(d1)

        return x4

class ResUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(ResUNet, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        # self.pool = nn.AdaptiveAvgPool2d(1)
        #self.Up5 = up_conv(ch_in=512, ch_out=256)
        #self.Up_conv5 = res_conv_block(ch_in=256, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        #self.Up_conv4 = res_conv_block(ch_in=128, ch_out=128)

        self.Up3 = up_conv(ch_in=256, ch_out=64)
        #self.Up_conv3 = res_conv_block(ch_in=64, ch_out=64)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        #self.Up_conv2 = res_conv_block(ch_in=64, ch_out=64)

        self.Up1 = up_conv(ch_in=128, ch_out=64)
        #self.Up_conv1 = res_conv_block(ch_in=64, ch_out=32)
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1)

    def forward(self, x):
        # encoding path  
        x0 = torch.cat([x,x,x],1)
        x0 = self.firstconv(x0)
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)
        x1 = self.firstmaxpool(x0)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)


        d4 = self.Up4(x4)
        d4 = torch.cat([x3, d4],dim=1)
        #d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat([x2, d3],dim=1)
        #d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat([x0, d2],dim=1)
        #d2 = self.Up_conv2(d2)

        d2 = self.Up1(d2)
        #d2 = self.Up_conv1(d2)

        d1 = self.Conv_1x1(d2)
        d1 = nn.Sigmoid()(d1)

        return  d1



class ResUNet1(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(ResUNet1, self).__init__()
        resnet = models.resnet34(pretrained=False)
        self.vgg = U_Net(img_ch=3)
        self.xception = Inception3()
        #self.encoder = Encoder()
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        # self.encoder4 = resnet.layer4
        # self.Up5 = up_conv(ch_in=512, ch_out=256)
        #self.Up_conv5 = res_conv_block(ch_in=256, ch_out=256)
        self.mean_head = nn.Sequential(nn.Conv2d(256, 256, 1),
                                       nn.BatchNorm2d(256),
                                       nn.LeakyReLU(0.2,inplace=True))
        self.logvar_head = nn.Sequential(nn.Conv2d(256, 256, 1),
                                       nn.BatchNorm2d(256),
                                       nn.LeakyReLU(0.2,inplace=True))
        self.mean_head1 = nn.Sequential(nn.Conv2d(256, 256, 1),
                                       nn.BatchNorm2d(256),
                                       nn.LeakyReLU(0.2, inplace=True))
        self.logvar_head1 = nn.Sequential(nn.Conv2d(256, 256, 1),
                                         nn.BatchNorm2d(256),
                                         nn.LeakyReLU(0.2, inplace=True))
        self.mean_head2 = nn.Sequential(nn.Conv2d(256, 256, 1),
                                       nn.BatchNorm2d(256),
                                       nn.LeakyReLU(0.2, inplace=True))
        self.logvar_head2 = nn.Sequential(nn.Conv2d(256, 256, 1),
                                         nn.BatchNorm2d(256),
                                         nn.LeakyReLU(0.2, inplace=True))
        self.mean_head3 = nn.Sequential(nn.Conv2d(256, 256, 1),
                                        nn.BatchNorm2d(256),
                                        nn.LeakyReLU(0.2, inplace=True))
        self.logvar_head3 = nn.Sequential(nn.Conv2d(256, 256, 1),
                                          nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.2, inplace=True))
        self.sk = SKConv(features=256,M=3,G=32,r=16,L=32)
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(256,2)
        # self.softmax = nn.Softmax()

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        # #self.Up_conv4 = res_conv_block(ch_in=128, ch_out=128)
        #
        self.Up3 = up_conv(ch_in=128, ch_out=64)
        # #self.Up_conv3 = res_conv_block(ch_in=64, ch_out=64)
        #
        self.Up2 = up_conv(ch_in=64, ch_out=64)
        # #self.Up_conv2 = res_conv_block(ch_in=64, ch_out=64)
        #
        self.Up1 = up_conv(ch_in=64, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1)

        #up_#
        # self.Up3_ = up_conv(ch_in=64, ch_out=16)
        # self.Up2_ = up_conv(ch_in=16, ch_out=4)
        # # self.Up1_ = up_conv(ch_in=4, ch_out=4)
        # self.encoder1_ = res_conv_block(stride=2,ch_in=4,ch_out=16)
        # self.encoder2_ = res_conv_block(stride=2, ch_in=16, ch_out=64)
        # self.encoder3_ = res_conv_block(stride=2, ch_in=64, ch_out=64)
        #self.Up_conv1 = res_conv_block(ch_in=64, ch_out=32)
        # self.fc1 = nn.Linear(256,128)
        # self.relu = nn.LeakyReLU(0.2,inplace=True)
        # self.dropout = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(128,64)

    def forward(self, x):
        # encoding path  
        x0_ = torch.cat([x,x,x],1)
        x0 = self.firstconv(x0_)
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)
        x1 = self.firstmaxpool(x0)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)

        # x4_1 = self.xception(x0_)
        # x4_2 = self.sk(x4,x4_,x4_1)
        # x5 = self.encoder4(x4)
        # up2_ = self.Up3_(x2)
        # up3_ = self.Up2_(up2_)
        # e4_ = self.encoder1_(up3_)
        # e3_ = self.encoder2_(e4_)
        # e2_ = self.encoder3_(e3_)
        mu0 = self.mean_head(x4)
        logvar0 = self.logvar_head(x4)
        # mu3 = self.mean_head2(x4_2)
        # logvar3 = self.logvar_head2(x4_2)
        mu_lst = [mu0]
        logvar_lst = [logvar0]




        if self.training:
           std = torch.exp(0.5*logvar0)
           eps = torch.randn_like(std)
           z = mu0 + std * eps
        else:
           z = mu0

           # encoding path
        d4 = self.Up4(z)


        d3 = self.Up3(d4)

        d2 = self.Up2(d3)


        d1 = self.Up1(d2)


        d0 = self.Conv_1x1(d1)
        d = nn.Sigmoid()(d0)




        return mu_lst, logvar_lst, z, d

# #########--------- Networks ---------#########



class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)
        
        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)
        
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)
        
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)
        
        d1 = self.Conv_1x1(d2)
        out = nn.Sigmoid()(d1)
        
        return out


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Conv_1x1(d2)
        out = nn.Sigmoid()(d1)
        
        return out



class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2AttU_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)
        
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)
        
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)
        
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)
        
        d1 = self.Conv_1x1(d2)
        out = nn.Sigmoid()(d1)
        
        return out
