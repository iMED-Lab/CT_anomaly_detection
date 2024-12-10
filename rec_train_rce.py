# -*- coding: utf-8 -*-
import torch
import argparse
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import imageio
from resnet import D_net,Vgg
from torch.utils.data import DataLoader
from dataset import CT_norm
from visualize import Visualizer
import os
import torch.nn as nn
from losses import dice_loss, kld_loss
from sklearn.metrics import roc_auc_score
from dis_loss import Dis_loss
import torch.optim as optim
from models import Get_model
import  torch.nn.functional as F
import time
def sigmoid(x):
    return 1 / (1 + np.exp(x))
 

def cross_entropy_error(p, y):
    """
 
    :param p: 预测结果
    :param y: 真值的 one-hot 编码
    :return:
    """
    delta = 1e-7  # 添加一个微小值，防止负无穷(np.log(0))的情况出现
    p = sigmoid(p)
    return -np.sum(y * np.log(p + delta) + (1 - y) * np.log(1 - p + delta))

class ReliableCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(ReliableCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true, reliability):
        # 计算交叉熵损失
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(y_pred, y_true)
        # print(loss.size())
        # th = torch.mean(reliability)
        # 将可靠性张量的形状调整为与标签相同
        reliability = reliability.view(-1)/reliability.max()
        
        # 使用可靠性加权损失
        weighted_loss = torch.mean(loss * reliability)
        
        return weighted_loss 

# class ReliableCrossEntropyLoss(nn.Module):
#     def __init__(self):
#         super(ReliableCrossEntropyLoss, self).__init__()

#     def forward(self, y_pred, y_true, reliability):
#         # 计算交叉熵损失
#         # criterion = nn.CrossEntropyLoss(reduction='none')
        
#         weights = reliability.view(-1)
        
#         y_pred = y_pred.view(-1)
#         y_true = y_true.view(-1)
#         print(weights.size(), y_pred.size(), y_true.size())


#         y_pred = torch.clamp(y_pred, min=1e-8, max=1-1e-8)

#     # 计算加权损失
#         loss = weights * (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    
#     # 返回损失的平均值（取负号）
#         return -torch.mean(loss)

        # reliability_post = reliability.clone().detach()
           # th = torch.mean(reliability)/50
        # reliability_post[reliability>=th]=0
        # reliability_post[reliability<th]=1
        # loss = criterion(y_pred, y_true + (reliability_post).long())
        # # 使用可靠性加权损失
        # weighted_loss = torch.mean(loss)
        # print(loss.size())
        # 将可靠性张量的形状调整为与标签相同
        # reliability = reliability.view(-1)
        
        # th = torch.mean(reliability)
        # y_true[reliability<0.01]=1
        # loss = criterion(y_pred, y_true)
        # # 使用可靠性加权损失
        # weighted_loss =  torch.mean(loss)
        # weighted_loss = torch.mean(loss * reliability)    
        # return weighted_loss 

def train_one_epoch(viz, encoder, decoder, embedding, train_dataloader, encoder_optimize, decoder_optimizer, embedding_optimizer, mse, rce, ce, device,epoch,max_epoch):
    epoch_loss = 0.0
    ce_loss_total = 0.0
    D_mean_loss = 0.0
    for i, (images, class_id) in enumerate(train_dataloader):
        img  = images.to(device)
        origin_image = img.clone().detach()
        # b, c, h, w = origin_image.size()
        # label_t = nn.Parameter(torch.ones(b,25,1,dtype=int),requires_grad=False).to(device)
        # label = nn.Parameter(torch.ones(b,231,1,dtype=int),requires_grad=False).to(device)
        start_time = time.time()
        x,_ = encoder(img)  
        quants_trunc, commitment_loss, vqem, codes = embedding(x)
        b,c,h,w = quants_trunc.size()
        # codes_lb = codes
        # codes_lb = codes_lb.view(b,int(h*w))
        # label_t = codes_lb[:,0:int(h*w*0.1)].reshape(-1)
        # label_t = F.one_hot(label_t,num_classes=2048)
        # # print(label_t.max(),label_t.min())
        # label_r = codes_lb[:,0:int(h*w*0.9+1)].reshape(-1)
        # # print(label_r.max(),label_r.min())
        # label_r = F.one_hot(label_r,num_classes=2048)
        ori_codes_t = torch.zeros(size=[b,int(h*w*0.1)],dtype=int).to(x.device)
        ori_codes_r = torch.ones(size=[b,int(h*w*0.9+1)],dtype=int).to(x.device)
        rec, post_fea, rec_trans, rec_trans_new, weight_var_t, out_q_t, out_q_r = decoder(img,x, quants_trunc,vqem,codes)
        end_time = time.time()
        print(end_time-start_time)
        # ce_loss_t = rce(out_q_t.view(-1,2), ori_codes_t.view(-1).detach(), weight_var_t.detach()) 
        # ce_loss_r = ce(out_q_r.view(-1,2), ori_codes_r.view(-1).detach()) 
        # ce_loss = ce_loss_t + ce_loss_r
        ce_loss_t = rce(out_q_t.view(-1,2), ori_codes_t.view(-1).detach(), weight_var_t.detach()) 
        ce_loss_t_w = ce(out_q_t.view(-1,2), ori_codes_t.view(-1).detach()) 
        ce_loss_r = ce(out_q_r.view(-1,2), ori_codes_r.view(-1).detach()) 
        weight_loss = min(1.0, 1 / (1 + math.exp(-5 * (epoch / max_epoch - 0.5))))
        # ce_loss = ce_loss_r + (weight_loss)*ce_loss_t + (1-weight_loss)*ce_loss_t_w
        ce_loss = ce_loss_r + ce_loss_t
        
        post_fea = torch.nn.Upsample(scale_factor=8,mode='nearest')(post_fea)
        res_fea = origin_image*(1-post_fea)
        res_fea_new = abs(origin_image-rec_trans_new)
        # D_real_decision = patch_GAN(rec).squeeze()
        # mini_batch = D_real_decision.size()
        # y_real_ = Variable(torch.ones(mini_batch).to(device))
        # y_fake_ = Variable(torch.zeros(mini_batch).to(device))
        # rec_trans_new_D = rec_trans_new.detach()
        # D_fake_decision = patch_GAN(rec_trans_new_D).squeeze()
        # D_fake_loss = mse(D_fake_decision, y_fake_)
        # D_real_loss = mse(D_real_decision, y_real_)
        # D_loss = D_real_loss + D_fake_loss
            # Back propagation
        # patch_GAN_optimizer.zero_grad()
        # D_loss.backward(retain_graph=True)
        # patch_GAN_optimizer.step()

        # D_mean_loss += (D_real_loss + D_fake_loss).item()
        # vis.plot(name='D_loss', y=D_mean_loss / (i + 1))
        
        # D_fake_decision = patch_GAN(rec_trans_new).squeeze()
        # D_loss2 = mse(D_fake_decision, y_real_)
        lossmse = mse(rec, origin_image)
        lossmse_r = mse(rec_trans_new,origin_image)
        # lossmse_r1 = mse(rec_r1,origin_image)
        # loss_t = mse(out_q_t,x_t.detach())
        # loss_r = mse(out_q_r, x_r.detach())
        # cross_entropy_t = ce(out_q_t.view(-1,2048),label_t.detach())
        # cross_entropy_r = ce(out_q_r.view(-1,2048),label_r.detach())
        # ce_loss = 0.99*cross_entropy_t+0.01*cross_entropy_r
        train_loss = 2*lossmse + commitment_loss + ce_loss + 2*lossmse_r
        # zero the parameter gradients
        # encoder_optimize.zero_grad()
        # patch_GAN_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # embedding_optimizer.zero_grad()
        # forward
        out_vis = torch.cat([img[0:4, 0:1, :, :],rec[0:4, 0:1, :, :],post_fea[0:4, 0:1, :, :],res_fea[0:4, 0:1, :, :], res_fea_new[0:4, 0:1, :, :], rec_trans[0:4, 0:1, :, :],rec_trans_new[0:4, 0:1, :, :]], 0)
        viz.img(name='train_vis', img_=out_vis, nrow=4)
        train_loss.backward()
        # encoder_optimize.step()
        decoder_optimizer.step()
        # embedding_optimizer.step()
        epoch_loss += train_loss.item()
        ce_loss_total += ce_loss.item()
        # pros, preds = pred.max(1)
        # print("%d / %d, train loss: %0.4f" % (i, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
        viz.plot("ce loss", ce_loss.item())
    print("epoch %d train_loss: %0.4f" % (epoch, epoch_loss/len(train_dataloader)))
    return encoder, decoder, embedding

def test(viz, encoder, decoder,embedding, test_dataloader, loss, device):
    epoch_loss = 0
    score = []
    score1 = []
    y_true = []
    for i, (images, class_id) in enumerate(test_dataloader):
        img  = images.to(device)
        label  = class_id.to(device)
        origin_image = img.clone().detach()
        b,c,h,w = origin_image.size()
        start_time = time.time()
        x,_ = encoder(img)  
        quants_trunc, commitment_loss, vqem, codes = embedding(x)
        rec, rec_new, post_fea = decoder.test(quants_trunc, vqem,codes)
        end_time = time.time()
        print(end_time-start_time)
        post_fea = torch.nn.Upsample(scale_factor=8)(post_fea)
        res_image = abs(origin_image-rec_new)
        new_post = post_fea*res_image
        pixel = new_post.view(b,-1).sum(dim=-1)
        num = torch.sum((new_post.view(b,-1))>0,dim=-1)
        score += (pixel/(num+1)).tolist()
        score1 += (pixel).tolist()
        y_true += label.tolist()
        test_loss = loss(rec, origin_image)
        
        # forward
        out_vis = torch.cat([img[0:4, 0:1, :, :], rec[0:4,0:1,:,:], rec_new[0:4,0:1,:,:], res_image[0:4,0:1,:,:], new_post[0:4, 0:1, :, :], post_fea[0:4, 0:1, :, :]], 0)
        viz.img(name='test_vis', img_=out_vis, nrow=4)
        epoch_loss += test_loss.item()
    score_new = [(val - min(score)) / (max(score) - min(score) + 0.00001) for val in score]
    score_new1 = [(val - min(score1)) / (max(score1) - min(score1) + 0.00001) for val in score1]
    print(len(score_new),len(y_true))
    auc = roc_auc_score(y_true,score_new)
    auc1 = roc_auc_score(y_true,score_new1)
    viz.plot("auc", auc)
    viz.plot("auc1", auc1)
    print(auc,auc1)

    return auc, auc1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default='/home/imed/personal/hhy/CT_detection/our_dataset_scan')
    # parser.add_argument("--test_domain",type=str, default={'ROSE-H'},help='OCTA500_3M, OCTA500_6M, ROSE-2, ROSE-O, ROSE-Z, ROSE-H')
    parser.add_argument("--gpu_id", type=str, default=[1], help="device")
    # parser.add_argument("--mode", type=str, default="train", choices=["train","val" "test"], help="train, val, test")
    parser.add_argument("--input_nc", type=int, default=3, choices=[1, 3], help="gray or rgb")
    # parser.add_argument("--scale_size", type=int, default=512, help="scale size")
    parser.add_argument("--vis_name", type=str, default='rqvae_rec_transformer_0.1_VQ_FC_AFS_1')

    parser.add_argument("--model_name", type=str, default='resnet18',help='resnet18, resnet34, resnet50, smt')
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="number of threads")
    parser.add_argument("--val_epoch_freq", type=int, default=1, help="frequency of validation at the end of epochs")
    parser.add_argument("--save_epoch_freq", type=int, default=5, help="frequency of saving models at the end of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--power", type=float, default=0.9, help="power")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")

    parser.add_argument("--training_epochs", type=int, default=200, help="train epochs of first stage")
    parser.add_argument("--models_dir", type=str, default="models", help="path to folder for saving models")
    parser.add_argument("--results_dir", type=str, default="results", help="path to folder for saving results")
    args = parser.parse_args()
    device = args.gpu_id
    # encoder = Get_model.Get_encoder().to(device[0])
    # encoder = Get_model.Get_encoder().to(device[0])
    # decoder = Get_model.Get_decoder().to(device[0])
    # embedding = Get_model.Get_embedding(size=32,channel=512,num = 1,share = True,n_embed=2048).to(device[0])
    embedding = torch.load('checkpoints/rqvae_rec_transformer/embedding-99.pt').to(device[0])
    encoder = torch.load('checkpoints/rqvae_rec_transformer/encoder-99.pt').to(device[0])
    decoder = Get_model.Get_decoder().to(device[0])
    # S = ResUNet1()
    # G = nn.DataParallel(G).cuda()
    # S = nn.DataParallel(S).cuda()
    # Loss function
    criterion1 = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss(reduce=True)
    criterion3 = torch.nn.MSELoss(reduce=True)
    criterion4 = dice_loss().cuda()
    criterion5 = torch.nn.MSELoss(reduce=True)
    criterion6 = torch.nn.MSELoss(reduce=True)
    criterion7 = kld_loss().cuda()
    criterion8 = torch.nn.CosineEmbeddingLoss(reduce=True)
    criterion9 = torch.nn.MSELoss(reduce=True)
    ce_loss = torch.nn.CrossEntropyLoss(reduce=True)
    rce_loss = ReliableCrossEntropyLoss()
    distance_loss = Dis_loss()

    # Optimizers
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    embedding_optimizer = torch.optim.Adam(embedding.parameters(), lr=args.lr)
    train_data_loader = DataLoader(dataset=CT_norm(root=args.base_dir,isTraining=True),
                                          batch_size=args.batch_size,
                                          shuffle=True)
    test_data_loader = DataLoader(dataset=CT_norm(root=args.base_dir,isTraining=False),
                                          batch_size=1,
                                          shuffle=False)
    # E_optimizer = torch.optim.Adam(E.parameters(), lr=learning_rate, betas=betas)
    # Training GAN
    # Fixed noise for test
    # num_test_samples = 5*5
    # fixed_noise = torch.randn(num_test_samples, G_input_dim).view(-1, G_input_dim, 1, 1)
    file_name = args.vis_name
    vis = Visualizer(env=file_name)
    best_auc = 0.0
    for epoch in range(args.training_epochs):
        # encoder.train()
        decoder.train()
        # embedding.train()
        encoder, decoder, embedding = train_one_epoch(viz=vis, encoder=encoder,decoder=decoder,embedding=embedding, train_dataloader=train_data_loader, \
                              encoder_optimize=encoder_optimizer, decoder_optimizer=decoder_optimizer, embedding_optimizer=embedding_optimizer,\
                              mse=mse_loss,rce=rce_loss, ce = ce_loss, device=device[0],epoch=epoch, max_epoch=args.training_epochs)
        if epoch % 1 == 0:
            with torch.no_grad():
                encoder.eval()
                decoder.eval()
                embedding.eval()
                auc = test(viz=vis, encoder=encoder,decoder=decoder,embedding=embedding, test_dataloader=test_data_loader,loss=mse_loss,device=device[0])
            checkpoint_path = os.path.join('checkpoints', args.vis_name)
            if os.path.exists(checkpoint_path) == False:
                  os.makedirs(checkpoint_path)
            checkpoint_path_1 = os.path.join(checkpoint_path, '{net}-{epoch}.pt')
            torch.save(encoder, checkpoint_path_1.format(net='encoder', epoch=epoch))
            torch.save(decoder, checkpoint_path_1.format(net='decoder', epoch=epoch))
            torch.save(embedding, checkpoint_path_1.format(net='embedding', epoch=epoch))

