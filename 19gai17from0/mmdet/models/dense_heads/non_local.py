import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, bias_init_with_prob, ConvModule,constant_init
from mmdet.core import multi_apply, bbox2roi, matrix_nms
from ..builder import HEADS, build_loss, build_head
from scipy import ndimage
import pdb
import matplotlib.pyplot as plt
from torch.nn import functional as F

@HEADS.register_module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

    def init_weights(self):
        pass


@HEADS.register_module
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.sum_after = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.human_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.human_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            self.human_phi = nn.Sequential(self.human_phi, max_pool_layer)

    def forward(self, feats_all=None,human_feats=None):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        query = feats_all
        key = feats_all
        value = feats_all
        batch_size = feats_all.size(0)
        human_query = self.human_theta(human_feats).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        human_key =self.human_phi(human_feats).view(batch_size, self.inter_channels, -1)
        human_map = torch.matmul(human_query, human_key)
        
        human_map = (self.inter_channels ** -.5) * human_map   

        human_map = F.softmax(human_map, dim=-1)
        # human_map=human_map.reshape(40,40,40,40)
        # for i in range(0,40,6):
        #     for j in range(0,40,6):
        #         plt.plot(j,i,'ks')
        #         plt.imshow(human_map[i][j].cpu().numpy())
        #         plt.show()
        
        

        g_x = self.g(value).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(query).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(key).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f = (self.inter_channels ** -.5) * f   
        f = F.softmax(f, dim=-1)


        # f=f.reshape(52,52,52,52)
        # human_map = human_map.reshape(52,52,52,52)
        # for i in range(0,52,3):
        #     for j in range(0,52,3):
                
        #         plt.subplot(1,3,1)
        #         plt.plot(j,i,'ks')
        #         plt.imshow(f[i][j].cpu().numpy())
        #         plt.subplot(1,3,2)
        #         plt.plot(j,i,'ks')
        #         plt.imshow((human_map)[i][j].cpu().numpy())
        #         plt.subplot(1,3,3)
        #         plt.plot(j,i,'ks')
        #         plt.imshow(((f+human_map)/2)[i][j].cpu().numpy())
        #         plt.show()


        f_div_C = (human_map+f)/2

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *feats_all.size()[2:])
        W_y = self.W(y)
        
        # for i in range(W_y.shape[1]):
        #     plt.subplot(1,3,1)
        #     plt.imshow(W_y[0][i].cpu().numpy())
        #     plt.subplot(1,3,2)
        #     plt.imshow(feats_all[0][i].cpu().numpy())
        #     plt.subplot(1,3,3)
        #     plt.imshow((W_y+feats_all)[0][i].cpu().numpy())
        #     plt.show()

        return W_y#+feats_all

    def minmaxscaler(self,data):
        amax=torch.max(data)
        amin=torch.min(data)
        return (data-amin)/(amax-amin)


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

@HEADS.register_module
class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
