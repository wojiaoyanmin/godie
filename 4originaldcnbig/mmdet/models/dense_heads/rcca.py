import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, bias_init_with_prob, ConvModule
from mmdet.core import multi_apply, bbox2roi, matrix_nms
from ..builder import HEADS, build_loss, build_head
from scipy import ndimage
import pdb
import matplotlib.pyplot as plt

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CC_module(nn.Module):
    def __init__(self, in_dim):
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().\
            view(m_batchsize * width, -1, height).permute(0, 2, 1)
        # n*c*h*w->n*w*c*h->nw*c*h->nw*h*c
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().\
            view(m_batchsize * height, -1, width).permute(0, 2,1)
        # n*c*h*w->n*h*c*w->nh*c*w->nh*w*c
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # n*c*h*w->n*w*c*h->nw*c*h
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        # n*c*h*w->n*h*c*w->nh*c*w
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # n*c*h*w->n*w*c*h->nw*c*h
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        # n*c*h*w->n*h*c*w->nh*c*w
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).\
            view(m_batchsize, width, height,height).permute(0,2,1,3)
        # bmm(nw*h*c  nw*c*h)+对角线-inf->nw*h*h->n*w*h*h->n*h*w*h
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        # bmm(nh*w*c  nh*c*w)->nh*w*w->n*h*w*w
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        # cat(n*h*w*h+n*h*w*w)->(n*h*w*(h+w))->softmax(dim3)
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # n*h*w*h->n*w*h*h->nw*h*h
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        # n*h*w*w->nh*w*w
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        # bmm(nw*c*h,nw*h*h(permute))->nw*c*h->n*w*c*h->n*c*h*w
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # bmm(nh*c*w,nh*w*w(permute))->nh*c*w->n*h*c*w->n*c*h*w
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x
        # gamma*(n*c*h*w+n*c*h*w)+h*c*h*w=

@HEADS.register_module
class RCCA(nn.Module):
    def __init__(self,
                 recurrence=2,
                 in_channels=256,
                 out_channels=256):
        super(RCCA, self).__init__()
        self.recurrence=recurrence
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.cc_module=CC_module(in_channels)

    def forward(self, feats, eval=False):
        for i in range(self.recurrence):
            feats=self.cc_module(feats)
        return feats


