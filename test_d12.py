import os
import cv2
import numpy as np
import skimage.io
# import skimage.viewer
# import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from timm.models.layers import DropPath, trunc_normal_
from torchvision import models
from torchvision import  transforms
from histoformer import *
from visformer import CAFormer
from datasets import *
from utils import *
from skimage import color

parser = argparse.ArgumentParser(description='histogram_network')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--epochs', type=int, default=100, help='the starting epoch count')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')

# args for Histoformer
parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
parser.add_argument('--embed_dim', type=int, default=12, help='dim of emdeding features')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='TwoDCFF', help='TwoDCFF/ffn token mlp')

parser.add_argument('--save_dir', type=str, default ='./checkpoints/',  help='save dir')
parser.add_argument('--save_image_dir', type=str, default ='./results_5_30/cube281d12/',  help='save image dir')
parser.add_argument('--outdir', type=str, default ='./results_txt/',  help='outdir dir')

parser.add_argument('--weight', type=str, default ='Histoformer_best.pth',  help='Histoformer weight')
# parser.add_argument('--weight_G', type=str, default ='Histoformer-PQR_netG_100.pth',  help='Generator weight')

opt = parser.parse_args()


def calc_mae(source, target):
  source = np.reshape(source, [-1, 3]).astype(np.float32)
  target = np.reshape(target, [-1, 3]).astype(np.float32)
  source_norm = np.sqrt(np.sum(np.power(source, 2), 1))
  target_norm = np.sqrt(np.sum(np.power(target, 2), 1))
  norm = source_norm * target_norm
  L = np.shape(norm)[0]
  inds = norm != 0
  angles = np.sum(source[inds, :] * target[inds, :], 1) / norm[inds]
  angles[angles > 1] = 1
  f = np.arccos(angles)
  f[np.isnan(f)] = 0
  f = f * 180 / np.pi
  return sum(f)/ (L)
def mean_angular_error(a, b):
    """Calculate mean angular error (via cosine similarity)."""
    return calc_mae(a, b)
model = Histoformer(embed_dim=opt.embed_dim,token_projection='linear',token_mlp='TwoDCFF')
net_u = CAFormer(embed_dim=(opt.embed_dim)*2,token_projection='linear').cuda()

model.to('cuda')   
# weight_path = r'D:\my_work\checkpoints\115mywork_5_30\cubed12\Histoformer_3_epoch_281.pth'
# weight_pathu = r'D:\my_work\checkpoints\115mywork_5_30\cubed12\CAformer_3_epoch_281.pth'
# checkpoint = torch.load(weight_path)
# checkpointu = torch.load(weight_pathu)
# # # 讀取模型權重
# model.load_state_dict(checkpoint['state_dict'])
# net_u.load_state_dict(checkpointu['state_dict'])
testloader= get_test_set()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
model.eval()
net_u.eval()
mse_lst, mae_lst, deltaE2000_lst = list(), list(), list()
with torch.no_grad():  #如果沒有這行，那下面在取值的時候要用.detach().numpy()
    for i,(inputs, ori_img,label_img,path) in enumerate(testloader):
        total_images = len(testloader.dataset)
        inputs = inputs.to(device)
        ori_img = ori_img.squeeze(0).numpy()
        ori_img_p = transforms.ToTensor()(ori_img).unsqueeze(0).to(device)
        label_img = label_img.squeeze(0).numpy()
        label_img = transforms.ToTensor()(label_img).unsqueeze(0).to(device)

        pred_img,cha_hist,hist_feaure = model(inputs)
        after_histo_image = padding_for_unet(ori_img_p)#bchw
        predicted_image = net_u(after_histo_image.float(),cha_hist,hist_feaure)
        
        if ori_img_p.size() != predicted_image.size():
            ori_a=ori_img_p.size()[2]
            ori_b=ori_img_p.size()[3]
            after_a=predicted_image.size()[2]
            after_b=predicted_image.size()[3]

            if ori_a != after_a:
                predicted_image = predicted_image[:, :, :-(after_a-ori_a),:]
            if ori_b != after_b:
                predicted_image = predicted_image[:, :, :, :-(after_b-ori_b)] 
        if ori_img_p.size() != predicted_image.size():
            print('incorrect')
        out_img = (torch.clamp(predicted_image,0,1))
        for gt_, out_ in zip(label_img.float().permute(0, 2, 3, 1), out_img.float().permute(0, 2, 3, 1)):

            mae_and_delta_Es = [[mean_angular_error(gt_.cpu().squeeze().numpy().reshape(-1, 3), out_.cpu().squeeze().numpy().reshape(-1, 3)),
                                    np.mean(calc_deltaE2000(cv2.cvtColor(gt_.cpu().squeeze().numpy(), cv2.COLOR_RGB2Lab), cv2.cvtColor(out_.cpu().squeeze().numpy(), cv2.COLOR_RGB2Lab)))]]
            mae, deltaE = np.mean(mae_and_delta_Es, axis=0)
            mse = (((gt_ - out_) * 255.) ** 2).mean().cpu().item()
            print("Sample {}/{}: MSE: {}, MAE: {}, DELTA_E: {}".format(i+1, total_images, mse, mae, deltaE), end="\n\n")
            sample_info = "Sample {}: MSE: {}, MAE: {}, DELTA_E: {}".format(path[0][38:], mse, mae, deltaE)
            with open(os.path.join(opt.outdir, f"{'HM30_cubesample281'}.txt"), "a") as f:
               f.write(sample_info+'\n')
            mse_lst.append(mse)
            mae_lst.append(mae)
            deltaE2000_lst.append(deltaE)
            print("Average:\n"
            "\nMSE: {}, Q1: {}, Q2: {}, Q3: {}"
            "\nMAE: {}, Q1: {}, Q2: {}, Q3: {}"
            "\nDELTA_E: {}, Q1: {}, Q2: {}, Q3: {}".format(np.mean(mse_lst), np.quantile(mse_lst, 0.25), np.quantile(mse_lst, 0.5), np.quantile(mse_lst, 0.75),
                                                            np.mean(mae_lst), np.quantile(mae_lst, 0.25), np.quantile(mae_lst, 0.5), np.quantile(mae_lst, 0.75),
                                                            np.mean(deltaE2000_lst), np.quantile(deltaE2000_lst, 0.25), np.quantile(deltaE2000_lst, 0.5), np.quantile(deltaE2000_lst, 0.75)))

        final_info = "\nFinal Info--->  \nMSE: {}, Q1: {}, Q2: {}, Q3: {} \nMAE: {}, Q1: {}, Q2: {}, Q3: {} \nDELTA_E: {}, Q1: {}, Q2: {}, Q3: {}".format(
        np.mean(mse_lst), np.quantile(mse_lst, 0.25), np.quantile(mse_lst, 0.5), np.quantile(mse_lst, 0.75),
        np.mean(mae_lst), np.quantile(mae_lst, 0.25), np.quantile(mae_lst, 0.5), np.quantile(mae_lst, 0.75),
        np.mean(deltaE2000_lst), np.quantile(deltaE2000_lst, 0.25), np.quantile(deltaE2000_lst, 0.5), np.quantile(deltaE2000_lst, 0.75))
        if not os.path.exists(opt.outdir):
            os.makedirs(opt.outdir)
        with open(os.path.join(opt.outdir, f"{'HM30_cube281'}.txt"), "w+") as f:
            f.write(final_info)

        # output_img = predicted_image.cpu().numpy().transpose((0, 2, 3, 1))
        # output_img = output_img[0, :, :, :]*255.0
        # output_img = np.array(output_img)
        # output_folder1 = os.path.join(opt.save_image_dir)
        # if not os.path.exists(output_folder1):
        #      os.makedirs(output_folder1)
        # print('path',path,path[0][38:])
        # cv2.imwrite(os.path.join(opt.save_image_dir, path[0][38:]), output_img ) #[34:] set2可能會要根據路徑的長度做更改 img_ori 
        ##原本是cv2.imwrite(os.path.join(opt.save_image_dir, ori_img[0][27:]), output_awb)38 cube