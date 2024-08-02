import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def L2_histo(x, y):
    bins = x.size(1)
    r = torch.arange(bins)
    s, t = torch.meshgrid(r, r)
    tt = t >= s
    tt = tt.to(device)

    cdf_x = torch.matmul(x, tt.float())
    cdf_y = torch.matmul(y, tt.float())

    return torch.sum(torch.square(cdf_x - cdf_y), dim=1)



class mae_loss():
    def compute(output, target):
        loss = torch.sum(torch.abs(output - target)) / output.size(0)
        return loss

    
