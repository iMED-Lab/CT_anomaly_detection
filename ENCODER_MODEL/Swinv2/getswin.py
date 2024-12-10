
from Models.ENCODER_MODEL.Swinv2.swin import SwinTransformerV2
import torch
def getswinv2(use_rq = False):
    model = SwinTransformerV2(img_size=512,
                                  patch_size=2,
                                  in_chans=3,
                                  num_classes=2,
                                  embed_dim=96,
                                  depths=[2, 2, 6, 2],
                                  num_heads=[3, 6, 12, 24],
                                  window_size=8,
                                  mlp_ratio=4.,
                                  qkv_bias=True,
                                  drop_rate= 0.0,
                                  drop_path_rate= 0.1,
                                  ape=False,
                                  patch_norm=True,
                                  use_checkpoint=False,
                                  pretrained_window_sizes=[0, 0, 0, 0],use_rq=use_rq)
    # checkpoint = torch.load('./pre_model/swinv2_tiny_patch4_window8_256.pth', map_location='cpu')
    # template_dict = model.state_dict()
    # state_dict = {k: v for k, v in checkpoint['model'].items() if k in template_dict and v.size() == template_dict[k].size()}
    
    # model.load_state_dict(state_dict, strict=False)
    return model
    
    
    



