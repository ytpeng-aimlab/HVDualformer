import torch.nn as nn
import torch
from torchvision.models import vgg16
from src import histoformer
from src import visformer
class VisNet(nn.Module):
    def __init__(self, inchnls=9,em_dim=16, device='cuda',wbset_num=3):
        """ Network constructor.
    """
        self.outchnls = int(inchnls / 3)
        self.inchnls = inchnls
        self.device = device
        super(VisNet, self).__init__()
        self.wbset_num = wbset_num
        self.net = visformer.CAFormer(embed_dim=(em_dim)*2,in_chans=inchnls,token_projection='linear',wbset_num=wbset_num).cuda().to(self.device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x,cha_hist,hist_feaure):
        """ Forward function"""
        # print('x',x.shape)#x torch.Size([1, 9, 128, 128])

        weights = self.net(x,cha_hist,hist_feaure)
        weights = torch.clamp(weights, -1000, 1000)
        # print('weights',weights.shape)# torch.Size([1, 3, 128, 128])
        weights = self.softmax(weights)
        out_img = torch.unsqueeze(weights[:, 0, :, :], dim=1) * x[:, :3, :, :]

        for i in range(1, int(self.wbset_num)):
            out_img += torch.unsqueeze(weights[:, i, :, :],
                                       dim=1) * x[:, (i * 3):3 + (i * 3), :, :]
        return out_img, weights

class HistNet(nn.Module):
    def __init__(self, inchnls=9,em_dim=16, device='cuda',wbset_num=3):

        self.inchnls = inchnls
        self.device = device
        super(HistNet, self).__init__()
        self.net = histoformer.Histoformer(in_chans=inchnls,embed_dim=em_dim,token_projection='linear',token_mlp='TwoDCFF',wbset_num=wbset_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """ Forward function"""
        out_hist,cha_hist,hist_feaure = self.net(x)
        return out_hist,cha_hist,hist_feaure
if __name__ == '__main__':
    x = torch.rand(8, 15, 64, 64).cuda()
    net = VisNet(15, 32)
    y, w = net(x)
