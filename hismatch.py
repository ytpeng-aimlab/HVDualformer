import torch
import random
import numpy as np

def HSFT(ca_image,his_image):
    B,L,C = ca_image.shape
    B1,L1,C1 = his_image.shape
    asli = ca_image.detach()
    asli_min, _ = torch.min(asli, dim=1, keepdim=True)  # Calculate the min along dim 1 (16384)
    # print('asli_min',asli_min.shape)
    asli = asli + (-asli_min)
    max0, _ = torch.max(asli, dim=1, keepdim=True)  # Calculate the min along dim 1 (16384)
    if torch.le(max0, 0).any() :
                asli = asli+0.0000000000000000001
                max0, _ = torch.max(asli, dim=1, keepdim=True)
    asli = asli / max0
    asli = torch.floor(torch.mul(asli, 255.0))
    # print('asli2',asli.shape,asli)
    result = asli.detach()
    input_tensor = (asli.to(torch.int64)).permute(0,2,1)
    histograms = torch.zeros((B, C, 256)).cuda()
    # 在特定维度上进行直方图统计
    histograms.scatter_add_(2, input_tensor, torch.ones_like(input_tensor, dtype=torch.float32))
    values = torch.arange(256, dtype=torch.uint8).unsqueeze(0).expand(B*C, -1).cuda()
    out_R = torch.nn.functional.softmax((his_image), dim=1)
    out_R = out_R.permute(0,2,1) ##bcl
    tensor = asli.permute(0,2,1).to(torch.int64)
    try:
         counts = torch.zeros((B, C, 256), dtype=torch.int32).cuda()
         indices = tensor
         counts.scatter_add_(-1, indices, torch.ones_like(indices, dtype=torch.int32))
         reshaped_input = input_tensor.reshape(B*C, L)
         unique_vals, idx = torch.unique(reshaped_input, return_inverse=True, sorted=False)
         o_quantiles_R = torch.cumsum(counts, dim=2, dtype=torch.float64)
         o_quantiles_R /= o_quantiles_R[:, :, -1].view(o_quantiles_R.size(0), o_quantiles_R.size(1), 1)
         sum_histograms = torch.sum(histograms, dim=2, keepdim=True)
         r_quantiles = torch.cumsum(torch.mul(out_R, sum_histograms), dim=2, dtype=torch.float64)
         r_quantiles /= r_quantiles[:, :, -1].view(r_quantiles.size(0), r_quantiles.size(1), 1)
         o_quantiles_R = o_quantiles_R.reshape(B*C,256)
         r_quantiles = r_quantiles.reshape(B*C,256)
         interp_t_valuesR = linear_interpolation_batch(o_quantiles_R,r_quantiles,values)
         result = (torch.gather(interp_t_valuesR, 1, idx)).reshape(B,C,L)
         result = result.permute(0,2,1)
         result= (((torch.floor(result)/255.0)*max0) - (-asli_min)).float()
    except ValueError:
        pass

    return result

def linear_interpolation_batch(x, xp, fp):
    # 寻找 x 在 xp 中的插值位置
    indices = torch.searchsorted(xp, x)

    # 处理超出范围的情况
    indices = torch.clamp(indices, 1, xp.size(1) - 1)

    # 计算插值权重
    t = (x - xp.gather(1, indices - 1)) / (xp.gather(1, indices) - xp.gather(1, indices - 1))

    # 执行一维线性插值
    result = fp.gather(1, indices - 1) + t * (fp.gather(1, indices) - fp.gather(1, indices - 1))

    return result.clamp(0, 255)



