import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init,bias_init_with_prob, ConvModule
from mmcv.cnn import CONV_LAYERS

import pdb


@CONV_LAYERS.register_module(name='ConvDW')
class ConvDW(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 norm_cfg=None):
        super(ConvDW, self).__init__(in_channels,
                                     out_channels,
                                     kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm_cfg = norm_cfg
        self.convs = nn.ModuleList()
        self.convs.append(
            ConvModule(
                self.in_channels,
                self.out_channels,
                1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                bias=self.norm_cfg is None))
        self.convs.append(
            ConvModule(
                self.out_channels,
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                groups=self.out_channels,
                norm_cfg=self.norm_cfg,
                bias=self.norm_cfg is None))
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.convs:
            normal_init(m.conv, std=0.01)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x
