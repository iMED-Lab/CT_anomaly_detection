from Models.ENCODER_MODEL.My_Net.hornet import HorNet
import torch
import torch.nn as nn
from Models.DECODER_MODEL.RES_DECODER import DecoderCup
import os
def get_hornet(pretrain = False):
    model = HorNet()
    # checkpoint = torch.load(os.getcwd()[:-17] + '/hornet_tiny_gf.pth',map_location='cpu')['model']
    # template_dict = model.state_dict()
    # state_dict = {k: v for k, v in checkpoint.items() if k in template_dict and v.size() == template_dict[k].size()}
    # if pretrain:
    #     model.load_state_dict(state_dict, False)
    return model
        