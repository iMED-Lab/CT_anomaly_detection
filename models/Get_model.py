import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.REC_MODEL.rqvae.models.rqvae.quantizations import RQBottleneck

def Get_embedding(size,channel,num = 4,share = True,n_embed=16000):
    embedding = RQBottleneck(latent_shape=[ size, size,channel ],code_shape=[size,size,num],
    n_embed=n_embed,decay=0.99,shared_codebook=share,restart_unused_codes=True)
    return embedding

def Get_encoder(in_channel=1):
    from models.ENCODER_MODEL.resnet.model import resnet18
    encoder = resnet18(pretrained=True,in_channel=in_channel)
    return encoder

# def Get_decoder(in_channels=[256,128,64],rqchannel=None,skip_channels=[0, 0,0,0],out_channels=1):
#     from models.DECODER_MODEL.res_decoder_v1 import decoder_main_encoder_VQ_FC
#     decoder = decoder_main_encoder_VQ_FC(in_channels=in_channels,rqchannel=rqchannel,skip_channels=skip_channels,out_channels=out_channels) 
#     return decoder
def Get_decoder(in_channels=[256,128,64],rqchannel=None,skip_channels=[0, 0,0,0],out_channels=1):
    from models.DECODER_MODEL.res_decoder_v1 import decoder_main_encoder_rce
    decoder = decoder_main_encoder_rce(in_channels=in_channels,rqchannel=rqchannel,skip_channels=skip_channels,out_channels=out_channels) 
    return decoder
    