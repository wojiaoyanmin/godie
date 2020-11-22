import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, kaiming_init, xavier_init
from ..builder import build_neck
from mmdet.core import auto_fp16
from ..builder import NECKS
from .fpn import FPN
import pdb

@NECKS.register_module()
class DoubleFpn(nn.Module):
    def __init__(self,
                 num_ints,
                 norm_cfg,
                 scale_factor_list,
                 Semantic_decoder,
                 Semantic_ASPP,
                 Cenoff_decoder):
        super(DoubleFpn, self).__init__()
        self.num_ints=num_ints
        self.norm_cfg=norm_cfg
        self.scale_factor_list=scale_factor_list
        self.Semantic_decoder = build_neck(Semantic_decoder)
        self.Semantic_ASPP=build_neck(Semantic_ASPP)
        self.Cenoff_decoder=build_neck(Cenoff_decoder)
        self.cenoff_inchannels=Cenoff_decoder['out_channels']
        self.sem_inchannels=Semantic_decoder['out_channels']
        self.__init__layers()
        self.init_weights()
    # default init_weights for conv(msra) and norm in ConvModule
    
    def __init__layers(self):
        self.cenoffs_conv=nn.ModuleList()
        for i in range(self.num_ints):
            self.cenoffs_conv.append(
                ConvModule(
                    self.cenoff_inchannels,
                    self.cenoff_inchannels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.sem_conv=ConvModule(
                self.sem_inchannels,
                self.sem_inchannels,
                3,
                stride=1,
                padding=1,
                norm_cfg=self.norm_cfg,
                bias=self.norm_cfg is None)

    def init_weights(self):
        self.Semantic_decoder.init_weights()
        self.Cenoff_decoder.init_weights()
        self.Semantic_ASPP.init_weights()
        for m in self.cenoffs_conv:
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, ConvModule):
                kaiming_init(m.conv)
        kaiming_init(self.sem_conv.conv)
    

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        outs_semantic=self.Semantic_decoder(inputs)[0]
        outs_semantic=F.interpolate(outs_semantic,
                                    scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)
        outs_semantic=self.sem_conv(outs_semantic)
        outs_semantic=self.Semantic_ASPP(outs_semantic)#(1,256,h/4,w/4)

        outs_cenoff=self.Cenoff_decoder(inputs)[:2]#[,,,]
        outs_cenoff=list(outs_cenoff)
        assert len(self.scale_factor_list)==self.num_ints,"len mismatch"
        for i in range(self.num_ints):
            outs_cenoff[i]=F.interpolate(outs_cenoff[i],
                                         scale_factor=self.scale_factor_list[i],
                                         mode='bilinear',
                                         align_corners=True)
        for i,conv in enumerate(self.cenoffs_conv):
            outs_cenoff[i]=conv(outs_cenoff[i])
        outs_cenoff=tuple(outs_cenoff)
        return outs_semantic,outs_cenoff#[1/4,1/8,1/16,1/16]