import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io
from skimage import io, transform
from os import path

from torch.utils.data import Dataset, DataLoader

from PIL import Image

pathcube = 'D:/FiFi_dataset/cuberender/CUBEimages/'
img_filescube = os.listdir(pathcube) #所有圖片的檔名
img_pathcube = [os.path.join("D:/FiFi_dataset/cuberender/CUBEimages/",i) for i in img_filescube]



def histogram_loader_BGR(path):
    image = cv2.imread(path)
    R_hist, R_bins = np.histogram(image[:, :, 2], bins=256, range=(0, 256)) # R_hist.shape = (256,)
    G_hist, G_bins = np.histogram(image[:, :, 1], bins=256, range=(0, 256))
    B_hist, B_bins = np.histogram(image[:, :, 0], bins=256, range=(0, 256))
    R_pdf = R_hist/sum(R_hist)
    G_pdf = G_hist/sum(G_hist)
    B_pdf = B_hist/sum(B_hist)
    BGR = np.vstack((B_pdf,G_pdf,R_pdf))
    return BGR
def get_test_set():
	test_data  = testset()#testset1 testset
	testloader = DataLoader(test_data, batch_size=1,shuffle=False)
	return testloader
#######cube#######

class testset(Dataset):
    def __init__(self):
        self.histogram_loader = histogram_loader_BGR

        self.images = img_pathcube #img_path6 #img_path6s
        self.gtdir = 'D:\\FiFi_dataset\\cuberender\\gt\\'
    def __getitem__(self, index):

        single_img = self.images[index]
        img_hist = self.histogram_loader(single_img)
        img_hist = torch.Tensor(img_hist)
        single_img_np = self.images[index]   
        file_name = os.path.splitext(os.path.split(single_img_np)[-1])[0]  # 得到 "1_AS"
        desired_part = file_name.split('_')[0]     
        input_img = cv2.imread(single_img_np) ###rgb 用io.read
        gt_filename = path.join(self.gtdir, desired_part + ".JPG")
        label_hist = self.histogram_loader(gt_filename)
        label_hist = torch.Tensor(label_hist)
        gt_img =  cv2.imread(gt_filename)
        return img_hist, input_img, gt_img,single_img_np

    def __len__(self):
        return len(self.images)
