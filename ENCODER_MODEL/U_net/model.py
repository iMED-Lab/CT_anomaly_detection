import torch
import torch.nn as nn

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

class Down_Sample(nn.Module):
    def __init__(self,in_dim=None, out_dim=None,padding=0):
        super(Down_Sample, self).__init__()
        self.block = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
    def forward(self,x):
        y = self.block(x)
        return y

class Up_Sample(nn.Module):
    def __init__(self,in_dim, out_dim,padding=0,output_padding=0):
        super(Up_Sample, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=(2, 2), stride=2, padding=padding,
                               output_padding=output_padding, bias=False),
            nn.ReLU(),
        )
    def forward(self,x):
        y = self.block(x)
        return y

class U_net_encoder(nn.Module):
    def __init__(self,in_chanels=3,rqchannel=None):
        super(U_net_encoder,self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Cov_block(in_chanels=3,med_chanels=32,out_chanels=32))
        self.layers.append(Cov_block(in_chanels=32,med_chanels=64,out_chanels=64))
        self.layers.append(Cov_block(in_chanels=64,med_chanels=128,out_chanels=128))
        self.layers.append(Cov_block(in_chanels=128,med_chanels=256,out_chanels=256))
        self.layers.append(Cov_block(in_chanels=256,med_chanels=512,out_chanels=512))
        self.down = nn.MaxPool2d(kernel_size=2,stride=2)
        self.rqchannel = rqchannel
        if self.rqchannel is not None:
            self.last = nn.Conv2d(in_channels=512,out_channels=rqchannel,kernel_size=1)
    def forward(self,x):
        feature = []
        for num,layer in enumerate(self.layers):
            x = layer(x)
            if num != len(self.layers)-1:
                feature.append(x)
                x = self.down(x)
        if self.rqchannel is not None:
            x = self.last(x)
        return x,feature

class U_net_decoder(nn.Module):
    def __init__(self,in_channels=[256, 128, 64,32],skip_channels=[256, 128, 64,32],out_channels=2,rqchannel = None):
        super(U_net_decoder,self).__init__()
        self.rqchannel = rqchannel
        if self.rqchannel is not None:
            self.first = nn.Conv2d(in_channels=rqchannel,out_channels=512,kernel_size=1)
        self.up = nn.ModuleList()
        for i in range(len(in_channels)):
            self.up.append(nn.ConvTranspose2d(in_channels=in_channels[i]*2,out_channels=in_channels[i],kernel_size=2,stride=2))
        self.stages = nn.ModuleList()
        for i in range(len(in_channels)):
            self.stages.append(Cov_block(in_chanels=in_channels[i]+skip_channels[i],med_chanels=in_channels[i],out_chanels=in_channels[i]))
        self.last = nn.Conv2d(in_channels=in_channels[-1],out_channels=out_channels,kernel_size=1)
    def forward(self, x, features=None):
        if features is not None:
            features = features[::-1]
        if self.rqchannel is not None:
            x = self.first(x)
        for i, decoder_block in enumerate(self.stages):
            if features is not None and i < len(features):
                skip = features[i]
            else:
                skip = None
            x = self.up[i](x)
            x = torch.cat((x,skip),dim=1) if skip is not None else x
            x = decoder_block(x)
        return self.last(x)
        

    
    