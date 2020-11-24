# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ..builder import HEADS, build_loss,build_head
import torch
from mmcv.cnn import normal_init,bias_init_with_prob, ConvModule
import torch.nn as nn
import pdb

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

class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h


@HEADS.register_module
class GloRe_Unit(nn.Module):
    """
    Graph-based Global Reasoning Unit

    Parameter:
        'normalize' is not necessary if the input size is fixed
    """

    def __init__(self, num_human,num_in, num_mid,
                 ConvNd=nn.Conv3d,
                 BatchNormNd=nn.BatchNorm3d,
                 normalize=False):
        super(GloRe_Unit, self).__init__()
        self.normalize = normalize
        self.num_human = num_human
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        self.down_channels = ConvModule(
                num_in,
                self.num_s*2,
                1,
                stride=1,
                padding=0,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True))
        # reduce dim
        self.conv_state = ConvNd(self.num_s*2, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(self.num_s*2, self.num_n, kernel_size=1)


        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # extend dimension
        
        self.recover = ConvNd(self.num_s *num_human, self.num_s, groups=self.num_s ,kernel_size=1, bias=False)
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1)
        self.blocker = BatchNormNd(num_in, eps=1e-04)  # should be zero initialized


    def forward(self, feats_all=None,human_pred=None):
        '''
        :param x: (n, c, d, h, w)

        '''
        batchsize,channels,h,w =feats_all.shape
        feat =self.down_channels(feats_all)
        human_pred=human_pred.unsqueeze(2)

        feat=feat.unsqueeze(1)
        feat=feat*human_pred
        
        feat=feat.reshape(batchsize*self.num_human,self.num_s*2,h,w)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)

        x_state_reshaped = self.conv_state(feat).view(batchsize*self.num_human, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(feat).view(batchsize*self.num_human, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(batchsize, self.num_human, self.num_s, *feat.size()[2:])
        
        x_state = x_state.permute(0,2,1,3,4)
        x_state = x_state.reshape(batchsize, self.num_s*self.num_human, *feat.size()[2:])
        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        x_state = self.recover(x_state)
        x_state = self.conv_extend(x_state)
        out = self.blocker(x_state)+feats_all

        return out

@HEADS.register_module
class GloRe_Unit_1D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_1D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv1d,
                                            BatchNormNd=nn.BatchNorm1d,
                                            normalize=normalize)

@HEADS.register_module
class GloRe_Unit_2D(GloRe_Unit):
    def __init__(self, num_human, num_in, num_mid, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_2D, self).__init__(num_human,num_in, num_mid,
                                            ConvNd=nn.Conv2d,
                                            BatchNormNd=nn.BatchNorm2d,
                                            normalize=normalize)

@HEADS.register_module
class GloRe_Unit_3D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_3D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv3d,
                                            BatchNormNd=nn.BatchNorm3d,
                                            normalize=normalize)


if __name__ == '__main__':

    for normalize in [True, False]:
        data = torch.autograd.Variable(torch.randn(2, 32, 1))
        net = GloRe_Unit_1D(32, 16, normalize)
        print(net(data).size())

        data = torch.autograd.Variable(torch.randn(2, 32, 14, 14))
        net = GloRe_Unit_2D(32, 16, normalize)
        print(net(data).size())

        data = torch.autograd.Variable(torch.randn(2, 32, 8, 14, 14))
        net = GloRe_Unit_3D(32, 16, normalize)
        print(net(data).size())
