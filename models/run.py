import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from Models.REC_MODEL.rqvae.models.rqvae.quantizations import RQBottleneck
from reconstruct.DARASET import dataset
from Pre_work import Pre_work,tools,losses
from torch.utils.data import DataLoader
from reconstruct import Get_model
import argparse
import torch

def get_parameter():
    Parameter = argparse.ArgumentParser()
    Parameter.add_argument('--model_name', default='ResNet18')  # 模型的名称
    Parameter.add_argument('--data_name', default='rec')  # 模型的名称
    Parameter.add_argument('--hyper', default='R+Z+O+A_4_book_Ori_contrastive_no_bili')
    Parameter.add_argument('--n_epochs', type=int, default=1200)  # 一共有几个epoch
    Parameter.add_argument('--batch_size', type=int, default=4)
    Parameter.add_argument('--lr', type=float, default=0.0001)
    Parameter.add_argument('--device', default='cuda:2')
    Parameter.add_argument('--image_size', type=int, default=512)
    Parameter.add_argument('--seed', type=int, default=1973)
    return Parameter.parse_args()
parameter = get_parameter()
use_diffusion = False if parameter.data_name == 'rec' else True
train_dataset,val_dataset = dataset(train_test='train',image_size = parameter.image_size,use_diffusion = use_diffusion),dataset(train_test='test',image_size = parameter.image_size,use_diffusion = use_diffusion)

train_dataloader = DataLoader(train_dataset, batch_size=parameter.batch_size, shuffle=True, num_workers=1)
val_dataloader = DataLoader(val_dataset, batch_size=parameter.batch_size, shuffle=False, num_workers=1)

encoder = Get_model.Get_encoder()
decoder = Get_model.Get_decoder()
embedding = Get_model.Get_embedding(size=32,channel=512,num = 4,share = True,n_embed=2048)

en_optimizer = torch.optim.AdamW(encoder.parameters(), lr=parameter.lr, weight_decay=0.0,betas=[0.5, 0.9])
de_optimizer = torch.optim.AdamW(decoder.parameters(), lr=parameter.lr, weight_decay=0.0,betas=[0.5, 0.9])
em_optimizer = torch.optim.AdamW(embedding.parameters(), lr=parameter.lr, weight_decay=0.0,betas=[0.5, 0.9])

display = tools.Display(epochs=parameter.n_epochs,out_dir='./rec_out/'+parameter.model_name+'/'+parameter.data_name + '/' + parameter.hyper + f'/{0}',fold = 0 )

train_matrix,val_matrix = tools.multi_matrix(num_class=1,train_test='train'),tools.multi_matrix(num_class=1,train_test='val')
device = torch.device(parameter.device)

encoder.to(device)
decoder.to(device)
embedding.to(device)

mse = torch.nn.MSELoss()

for epoch in range(parameter.n_epochs):
    all_loss_mse,all_loss_vq,all_loss_contras = 0.0, 0.0,0.0
    
    for batch in train_dataloader:
        encoder.train()
        decoder.train()
        embedding.train()
        en_optimizer.zero_grad()
        de_optimizer.zero_grad()
        em_optimizer.zero_grad()
        # print(batch['image'])
        image = batch['image'].to(device)
        
        label = batch['label'].to(device)
        name = batch['name']

        origin_image = image.clone().detach()

        x,_ = encoder(image)  
        # print(x.shape)
        quants_trunc, commitment_loss,_,constrastive_loss = embedding(x)
        rec_mask = decoder(quants_trunc)

        lossmse = mse(rec_mask, origin_image)

        # loss = lossmse + 0.25 * commitment_loss + 0.25 * constrastive_loss
        loss = lossmse +  commitment_loss + constrastive_loss

        loss.backward()
        en_optimizer.step()
        de_optimizer.step()
        em_optimizer.step()

        all_loss_mse += lossmse.cpu().item()
        all_loss_vq += commitment_loss.cpu().item()
        all_loss_contras += constrastive_loss.cpu().item()
        
        pred = torch.cat((rec_mask,-rec_mask),dim=1)
        train_matrix.calculate_score(pred=pred,label=label)
        tools.save_image(image = image.cpu().detach(),pred=pred.cpu().detach(),label=label.cpu().detach(),name = name, out_dir = './rec_out/'+parameter.model_name+'/'+parameter.data_name + '/' + parameter.hyper + f'/{0}',save_dir = 'train_image')

    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        embedding.eval()
        for batch in val_dataloader:
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            name = batch['name']

            x,_ = encoder(image)  
            quants_trunc, commitment_loss,_,constrastive_loss = embedding(x)
            rec_mask = decoder(quants_trunc)

            pred = torch.cat((rec_mask,-rec_mask),dim=1)

            val_matrix.calculate_score(pred=pred,label=label)
            tools.save_image(image = image.cpu().detach(),pred=pred.cpu().detach(),label=label.cpu().detach(),name = name,
                             out_dir = './rec_out/'+parameter.model_name+'/'+parameter.data_name + '/' + parameter.hyper + f'/{0}',save_dir = 'val_image')
    train_score,val_score = train_matrix.get_score(),val_matrix.get_score()
    index = tools.get_index(val_score)
    display.save_all(stict={'encoder':encoder.state_dict(),'decoder':decoder.state_dict(),'embedding':embedding.state_dict()},index=index)
    display.display(epoch=epoch + 1, losses={'Loss': 'Loss', 'MSE_LOSS': all_loss_mse,'RQ_Loss':all_loss_vq,'Contrastive_Loss':all_loss_contras},
                        scores=[train_score,val_score], images=None, cal_best=[index, val_score],
                        hyper=parameter.hyper)
tools.mask_json(dict=display.best_score,out_dir='./rec_out/'+parameter.model_name+'/'+parameter.data_name + '/' + parameter.hyper + f'/{0}')