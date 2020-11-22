import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, kaiming_init, xavier_init

from ..builder import NECKS
import pdb

@NECKS.register_module()
class ASPP(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling)

    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
    """

    def __init__(self,
                 in_channels=2048,
                 out_channels=256,
                 atrous_rates=(1, 6, 12, 18, 1),
                 norm_cfg=None):
        super().__init__()
        self.in_channels=in_channels#2048
        self.out_channels=out_channels#256
        self.atrous_rates = atrous_rates#(1, 6, 12, 18, 1)
        self.norm_cfg=norm_cfg
        self.aspp=nn.ModuleList()
        for atrous_rate in atrous_rates:
            kernel_size = 3 if atrous_rate > 1 else 1
            padding = atrous_rate if atrous_rate > 1 else 0
            #norm=None if i==len(atrous_rates)-1 else self.norm_cfg 
            conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=atrous_rate,
                padding=padding,
                norm_cfg=self.norm_cfg,
                bias=self.norm_cfg is None)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.project=ConvModule(
            5 * out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=self.norm_cfg,
            bias=self.norm_cfg is None )
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1,padding=0, bias=False),#1280,256
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, ConvModule):
                kaiming_init(m.conv)


    def forward(self, x):
        
        avg_x = self.gap(x)#torch.Size([1, 256, 1, 1])
        out = []
        for aspp_idx in range(len(self.aspp)):#len=4
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(self.aspp[aspp_idx](inp))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        out=self.project(out)
        return out
