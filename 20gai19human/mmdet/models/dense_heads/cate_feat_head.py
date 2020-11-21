import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init,bias_init_with_prob, ConvModule

from ..builder import HEADS, build_loss,build_head,build_neck

import torch
import numpy as np
import pdb


@HEADS.register_module
class CateFeatHead(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 num_ints=5,
                 conv_cfg=None,
                 norm_cfg=None,
                 stack_convs=4):
        super(CateFeatHead, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_ints = num_ints
        self.stack_convs=stack_convs

        self.lateral_convs=nn.ModuleList()
        for i in range(self.num_ints):
            self.lateral_convs.append(
                ConvModule(
                    self.in_channels,
                    self.mid_channels,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        self.convs = nn.ModuleList()
        for i in range(self.stack_convs):
            chn=self.mid_channels+2 if i==0 else self.mid_channels 
            self.convs.append(
                ConvModule(
                    chn,
                    self.mid_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None
                    ))

        self.pred=nn.Conv2d(
            self.mid_channels,
            self.out_channels, 
            1, 
            padding=0)

        self.feature_conv = ConvModule(
            self.mid_channels,
            self.in_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            act_cfg=None,
            norm_cfg=self.norm_cfg,
            bias=self.norm_cfg is None)

    def init_weights(self):
        for m in self.lateral_convs:
            normal_init(m.conv, std=0.01)
        for m in self.convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.pred, std=0.01)

    def forward(self, inputs):
        size=inputs[1].shape[-2:]
        assert self.num_ints==len(self.lateral_convs)
        feats_all = []
        for conv, feat in zip(self.lateral_convs, inputs):
            feat = conv(F.interpolate(feat, size=size, mode='bilinear', align_corners=True)).unsqueeze(0)
            feats_all.append(feat)
        feats_all = torch.sum(torch.cat(feats_all, dim=0), dim=0)
        x_range = torch.linspace(-1, 1, feats_all.shape[-1], device=feats_all.device)
        y_range = torch.linspace(-1, 1, feats_all.shape[-2], device=feats_all.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([feats_all.shape[0], 1, -1, -1])
        x = x.expand([feats_all.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        feats_all = torch.cat([feats_all, coord_feat], 1)
        for conv in self.convs:
            feats_all =conv(feats_all)
        feature_pred = self.pred(feats_all)
        feats_all =self.feature_conv(feats_all)
        return feats_all,feature_pred