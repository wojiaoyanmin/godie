import torch
import pdb
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import normal_init, bias_init_with_prob, ConvModule, constant_init
from ..builder import HEADS, build_loss, build_head, build_neck
import matplotlib.pyplot as plt


class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x, A):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        x= x-torch.matmul(A, x)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h

@HEADS.register_module
class YMLocal(nn.Module):
    def __init__(self, recurrence,  in_channels, key_channels=None, norm_cfg=None):
        super(YMLocal, self).__init__()

        self.recurrence = recurrence
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.norm_cfg = norm_cfg
        if self.key_channels is None:
            self.key_channels = in_channels // 2
            if self.key_channels == 0:
                self.key_channels = 1
        self.f_query = nn.Conv2d(
            self.in_channels,
            self.key_channels,
            1,
            padding=0)

        self.f_key = nn.Conv2d(
            self.in_channels,
            self.key_channels,
            1,
            padding=0)

        self.f_down = nn.Conv2d(
            self.in_channels,
            self.key_channels,
            1,
            padding=0)
        self.f = ConvModule(
            self.key_channels,
            self.key_channels,
            1,
            stride=1,
            padding=0,
            norm_cfg=self.norm_cfg,
            bias=self.norm_cfg is None)

        self.f_up = ConvModule(
            self.key_channels,
            self.in_channels,
            1,
            stride=1,
            padding=0,
            norm_cfg=self.norm_cfg,
            act_cfg=None,
            bias=self.norm_cfg is None)

    def init_weights(self):
        pass

    def forward(self, x ,human):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        value = x
        query=human
        key=human
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        query = self.f_query(query)
        query = query.view(batch_size, self.key_channels, -1)

        # key= self.f_key(key)
        # key = key.view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query.permute(0, 2, 1), query)
        sim_map = (self.key_channels ** -.5) * sim_map

        value = self.f_down(value).view(batch_size, self.key_channels, -1)
        ## reasoning: (n, num_state, num_node) -> (n, num_state, num_node)

        value = value.permute(0, 2, 1)

        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)

        context = context.permute(0, 2, 1).contiguous()

        context = context.view(batch_size, self.key_channels, h, w)
        context =self.f(context)
        context = self.f_up(context)
        x = x + context
        return x





