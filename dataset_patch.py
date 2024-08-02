import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from os.path import join
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import random
from scipy.io import loadmat

def histogram_loaderpatch_BGR(image):
    R_hist, R_bins = np.histogram(image[:, :, 2], bins=256, range=(0, 256)) # R_hist.shape = (256,)
    G_hist, G_bins = np.histogram(image[:, :, 1], bins=256, range=(0, 256))
    B_hist, B_bins = np.histogram(image[:, :, 0], bins=256, range=(0, 256))

    R_pdf = R_hist/sum(R_hist)
    G_pdf = G_hist/sum(G_hist)
    B_pdf = B_hist/sum(B_hist)
    BGR = np.vstack((B_pdf,G_pdf,R_pdf))
    return BGR

class BasicDataset_BGR(Dataset):
    def __init__(self, imgs_dir, fold=0, patch_size=128, patch_num_per_image=1, max_trdata=12000):

        self.imgs_dir = imgs_dir
        self.patch_size = patch_size
        self.patch_num_per_image = patch_num_per_image
        # get selected training data based on the current fold

        if fold is 0:
            logging.info(f'Training process will use {max_trdata} training images randomly selected from all training data')
            logging.info('Loading training images information...')
            self.imgfiles = [join(imgs_dir, file) for file in listdir(imgs_dir)
                        if not file.startswith('.')]
        else:
            logging.info(f'There is no fold {fold}! Training process will use all training data.')

        if max_trdata is not 0 and len(self.imgfiles) > max_trdata:
            print('>12000')
            random.shuffle(self.imgfiles)
            self.imgfiles = self.imgfiles[0:max_trdata]
            # for i in range(len(self.imgfiles)):
            #     with open(os.path.join('./checkpoints/mywork1', 'paperdataset.txt'), 'a') as f:
            #        f.write(str(self.imgfiles[i])+'\n')
            #        f.close()
        logging.info(f'Creating dataset with {len(self.imgfiles)} examples')

    def __len__(self):
        return len(self.imgfiles)

    @classmethod
    def preprocess(cls, pil_img, patch_size, patch_coords, flip_op):
        if flip_op is 1:
            pil_img = np.flip(pil_img, axis=1)
        elif flip_op is 2:
            pil_img = np.flip(pil_img,axis=0)

        img_nd = np.array(pil_img)
        assert len(img_nd.shape) == 3, 'Training/validation images should be 3 channels colored images'
        img_nd = img_nd[patch_coords[1]:patch_coords[1]+patch_size, patch_coords[0]:patch_coords[0]+patch_size, :]
        # print('img_nd.shape',img_nd.shape)
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans
        # print('img_trans.shape',img_trans.shape)

        return img_trans

    def __getitem__(self, i):
        gt_ext = ('G_AS.png','G_AS.png')
        img_file = self.imgfiles[i]
        in_img = cv2.imread(img_file)
        # print('img_file',img_file,in_img.shape)

        # get image size
        w, h = in_img.shape[1],in_img.shape[0]
        # get ground truth images
        parts = img_file.split('_')
        base_name = ''
        for i in range(len(parts) - 2):
            base_name = base_name + parts[i] + '_'
        gt_awb_file = base_name + gt_ext[0]
        # print('gt_awb_file',gt_awb_file)
        parts = gt_awb_file.split('\\')
        # print(parts)
        parts[-2] = 'gt'
        gt_awb_file = '\\'.join(parts)
        # print(gt_awb_file)
        awb_img = cv2.imread(gt_awb_file)
        # get flipping option
        flip_op = np.random.randint(3)
        # get random patch coord
        patch_x = np.random.randint(0, high=w - self.patch_size)
        patch_y = np.random.randint(0, high=h - self.patch_size)
        in_img_patches = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
        img_patches = in_img_patches.transpose((1,2,0)) #hwc
        img_patches_hist = histogram_loaderpatch_BGR(img_patches)
        awb_img_patches = self.preprocess(awb_img, self.patch_size, (patch_x, patch_y), flip_op)
        gt_patches = awb_img_patches.transpose((1,2,0))
        gt_patches_hist = histogram_loaderpatch_BGR(gt_patches)
        


        for j in range(self.patch_num_per_image - 1):
            # get flipping option
            flip_op = np.random.randint(3)
            # get random patch coord
            patch_x = np.random.randint(0, high=w - self.patch_size)
            patch_y = np.random.randint(0, high=h - self.patch_size)
            temp = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
            temp_img = temp.transpose((1,2,0))
            temp_hist = histogram_loaderpatch_BGR(temp_img)
            in_img_patches = np.append(in_img_patches, temp, axis=0)
            img_patches_hist = np.append(img_patches_hist, temp_hist, axis=0)
            temp = self.preprocess(awb_img, self.patch_size, (patch_x, patch_y), flip_op)
            temp_gt = temp.transpose((1,2,0))
            temp_gt_hist = histogram_loaderpatch_BGR(temp_gt)
            gt_patches_hist = np.append(gt_patches_hist, temp_gt_hist, axis=0)
            awb_img_patches = np.append(awb_img_patches, temp, axis=0)

        return {'image': (in_img_patches), 'gt-AWB': (awb_img_patches),'image_hist': (img_patches_hist),'gt_hist': (gt_patches_hist)}

