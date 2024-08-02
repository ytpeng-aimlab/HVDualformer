import os
import cv2
import numpy as np
import skimage.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from torch.autograd import Variable
from timm.models.layers import DropPath, trunc_normal_
from torchvision import models
from torchvision import  transforms

from dataset_patch import *

from datasets import *
from utils import *

from loss import *
from histoformer import *
from visformer import CAFormer

# from model_intraMLP import *
def main(opt):
    dataset = BasicDataset_BGR(opt.dir_img, fold=0, patch_size=opt.patch_size, patch_num_per_image=4, max_trdata=12000)
    trainloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,num_workers=8, pin_memory=True)

    ### Model ###
    model = Histoformer(embed_dim=opt.embed_dim,token_projection='linear',token_mlp='TwoDCFF').cuda()
    net_u = CAFormer(embed_dim=(opt.embed_dim)*2,token_projection='linear').cuda()

    ### Cuda/GPU ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### Optimizer ###
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay) #
    optimizer_u = torch.optim.Adam(net_u.parameters(), lr=opt.lr, betas=(0.5, 0.999))

#     checkpoint_model = torch.load(r'D:\my_work\checkpoints\mywork_5_30\Histoformer_epoch_.pth') 
#     model.load_state_dict(checkpoint_model['state_dict'])
#     optimizer.load_state_dict(checkpoint_model['optimizer'])
# ##############
#     checkpoint_net_u = torch.load(r'D:\my_work\checkpoints\mywork_5_30\CAformer_epoch_.pth')
#     net_u.load_state_dict(checkpoint_net_u['state_dict'])
#     optimizer_u.load_state_dict(checkpoint_net_u['optimizer'])

    patchnum=4
    ### Loss ###
    criterionL1 = nn.L1Loss().cuda()
    ### Summary Writer Tensorboard ###
    writer = SummaryWriter(os.path.join(opt.save_dir, 'tensorboard'))

    import sys
    iteration=1
    for e in range(opt.epochs):
        torch.cuda.empty_cache()
        model.train()
        net_u.train()
        for ik,(batch) in enumerate(trainloader):
            imgs_ = batch['image']
            awb_gt_ = batch['gt-AWB']
            # print('awb_gt_',awb_gt_.size())
            imgs_hist = batch['image_hist']
            # print('imgs_hist',imgs_hist.size())
            awb_gt_hist = batch['gt_hist']
            # print('awb_gt_hist',awb_gt_hist.size())
            for j in range(patchnum):
                input_img = imgs_[:, (j * 3): 3 + (j * 3), :, :]  #(1, 3, 128, 128)
                label_img= awb_gt_[:, (j * 3): 3 + (j * 3), :, :]
                img_hist = imgs_hist[:, (j * 3): 3 + (j * 3), :]
                label_hist=awb_gt_hist[:, (j * 3): 3 + (j * 3), :]

                input_hist = img_hist.to(torch.float).to(device)
                label_hist = label_hist.to(torch.float).to(device)
                input_img = input_img.to(device)  #(1, 3, 128, 128)
                label_img = label_img.to(device)  #(1, 3, 128, 128)


                optimizer.zero_grad()
                pred_hist,cha_hist,hist_feaure  = model(input_hist)
                # print('pred_img',pred_hist.size())

                B_out = pred_hist[:,0]
                # print('R_out',R_out.size())
                G_out = pred_hist[:,1]
                R_out = pred_hist[:,2]
                B_labels = label_hist[:,0]
                G_labels = label_hist[:,1]
                R_labels = label_hist[:,2]
                R_loss = L2_histo(R_out,R_labels)
                G_loss = L2_histo(G_out,G_labels)
                B_loss = L2_histo(B_out,B_labels)

                input_img = (input_img/255.0)
                gt = (label_img/255.0)
                # print('gt',gt.shape)

                RGB_loss = (1*R_loss)+(1*G_loss)+(1*B_loss)
                loss = torch.mean(RGB_loss)
                loss.backward()
                optimizer.step()             
                writer.add_scalar('loss_HISTO_OUTPUT', loss,iteration)

                predicted_image = net_u(input_img,cha_hist,hist_feaure)
                # print('predicted_image',predicted_image.shape)
                optimizer_u.zero_grad()        
                maeloss_u = criterionL1(predicted_image, gt)
                loss_u = maeloss_u
                loss_u.backward(retain_graph=True) #retain_graph=True

                optimizer_u.step()
                        
                    #####################
                loss_sum =  loss + loss_u #
                # loss = loss
                    
                writer.add_scalar('loss_u', loss_u, iteration) 
                writer.add_scalar('loss_sum', loss_sum, iteration) 
                iteration+=1
            if ik %20==0:
                print('epoch: {} , batch: {}, losssum: {},  loss_u: {}'.format(e + 1, ik + 1, loss_sum.data,loss_u.data))

        if (e+1)>250 or (e+1)%10==0: 
                torch.save({'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(opt.save_dir,"Histoformer_epoch_{}.pth".format(e+1)))         
                torch.save({'state_dict': net_u.state_dict(),
                'optimizer' : optimizer_u.state_dict()
                }, os.path.join(opt.save_dir,"CAformer_epoch_{}.pth".format(e+1)))       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='histogram_network')
    # global settings
    
    parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--epochs', type=int, default=350, help='the starting epoch count')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
    parser.add_argument('--resume', type=str, default ='',  help='save image dir')
    parser.add_argument('--patch_size', type=int, default=128, help='training patch_size')

    # args for Histoformer
    parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
    parser.add_argument('--embed_dim', type=int, default=8, help='dim of emdeding features')
    parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
    parser.add_argument('--token_mlp', type=str,default='TwoDCFF', help='TwoDCFF/ffn token mlp')
    parser.add_argument('--save_dir', type=str, default ='./checkpoints/mywork_5_30d8',  help='save dir')
    parser.add_argument('--save_image_dir', type=str, default ='./results/',  help='save image dir')
    parser.add_argument('--dir_img', type=str, default =r'D:\my_work\datasets\train\input',  help='read dir_img path')

    opt = parser.parse_args()
    print (opt)
    main(opt)