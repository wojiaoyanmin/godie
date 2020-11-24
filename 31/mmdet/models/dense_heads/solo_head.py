import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init,bias_init_with_prob, ConvModule
from mmdet.core import multi_apply, bbox2roi, matrix_nms, multiclass_nms
from ..builder import HEADS, build_loss,build_head
from scipy import ndimage
import pdb
import matplotlib.pyplot as plt
from mmdet.ops import ConvDW
import os.path as osp
import numpy as np
INF = 1e8
import cv2

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep



@HEADS.register_module
class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes=19,  # 不算background
                 in_channels=256,
                 stacked_convs=4,
                 cate_stacked_convs=5,
                 cate_stacked_convs_after=2,
                 seg_stacked_convs=4,
                 ins_feat_channels=512,
                 cate_feat_channels=512,
                 strides=None,
                 scale_ranges=((1, 48), (24, 96), (48, 192), (96, 2048)),
                 grid_big=40,
                 sigma=0.2,
                 num_grids=None,
                 ins_out_channels=None,
                 cate_feat_head=None,
                 aspp=None,
                 reasoning=None,
                 sa=None,
                 loss_ins=None,
                 loss_cate=None,
                 loss_human=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SOLOHead, self).__init__()
        self.sa_list=nn.ModuleList()
        for i in range(len(num_grids)):
            self.sa_list.append(build_head(sa))
        self.reasoning=build_head(reasoning)
        self.num_classes = num_classes  # 不算background
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.cate_stacked_convs =cate_stacked_convs
        self.cate_stacked_convs_after = cate_stacked_convs_after
        self.seg_stacked_convs = seg_stacked_convs
        self.ins_feat_channels = ins_feat_channels
        self.cate_feat_channels = cate_feat_channels
        self.strides = strides
        self.scale_ranges = scale_ranges
        self.grid_big = grid_big
        self.sigma = sigma
        self.seg_num_grids = num_grids
        self.ins_out_channels = ins_out_channels
        self.kernel_out_channels = ins_out_channels*1*1
        self.cate_feat_head = build_head(cate_feat_head)
        self.loss_human = build_loss(loss_human)
        self.loss_ins = build_loss(loss_ins)
        self.loss_cate = build_loss(loss_cate)
        self.conv_cfg=conv_cfg
        self.norm_cfg = norm_cfg
        assert len(self.seg_num_grids)==len(self.strides)==len(self.scale_ranges)
        self._init_layers()

    def _init_layers(self):
     
        cfg_conv = self.conv_cfg
        norm_cfg = self.norm_cfg
        self.cate_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            
            chn = self.in_channels + 2 if i == 0 else self.ins_feat_channels
            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.ins_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

        for i in range(self.cate_stacked_convs):
            chn = self.in_channels if i == 0 else self.cate_feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.cate_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

        self.cate_convs_after = nn.ModuleList()
        for i in range(self.cate_stacked_convs_after):
            self.cate_convs_after.append(
                ConvModule(
                    self.cate_feat_channels,
                    self.cate_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

        self.lateral_convs=nn.ModuleList()
        for i in range(len(self.seg_num_grids)):
            self.lateral_convs.append(
                ConvModule(
                    self.cate_feat_channels,
                    self.cate_feat_channels,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

        self.all_conv = ConvModule(
                    self.cate_feat_channels,
                    self.cate_feat_channels,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None)


        self.solo_cate = nn.Conv2d(
            self.cate_feat_channels, self.num_classes, 3, padding=1)

        self.solo_kernel = nn.Conv2d(
            self.ins_feat_channels, self.kernel_out_channels, 3, padding=1)

    def init_weights(self):
        for m in self.sa_list:
            m.init_weights()
        self.cate_feat_head.init_weights()
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cate_convs_after:
            normal_init(m.conv, std=0.01)
        for m in self.lateral_convs:
            normal_init(m.conv, std=0.01)
        for m in self.kernel_convs:
            normal_init(m.conv, std=0.01)
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)
        normal_init(self.solo_kernel, std=0.01)
        normal_init(self.all_conv.conv, std=0.01)

    def forward(self, feats,img_metas=None, eval=False):
        '''
        print(img_metas[0]['filename'])
        filename=img_metas[0]['filename']
        filename=osp.join(osp.join(osp.dirname(osp.split(filename)[0]),'Instances'),osp.split(filename)[1][:-3]+'png')
        img=plt.imread(filename)
        showimg=[]
        for feat in feats:
            showimg.append(F.interpolate(feat,size=feats[0].size()[-2:],mode='bilinear'))
        
        plt.subplot(2,3,6)
        plt.imshow(img)
        plt.subplot(2,3,1)
        plt.imshow(torch.max(showimg[0][0],0)[0].detach().cpu().numpy())
        plt.subplot(2,3,2)
        plt.imshow(torch.max(showimg[1][0],0)[0].detach().cpu().numpy())
        plt.subplot(2,3,3)
        plt.imshow(torch.max(showimg[2][0],0)[0].detach().cpu().numpy())
        plt.subplot(2,3,4)
        plt.imshow(torch.max(showimg[3][0],0)[0].detach().cpu().numpy())
        plt.show()
        '''
        human_feats,human_pred = self.cate_feat_head(feats[:-1])
        human_pred = human_pred.sigmoid()
        feats = self.split_feats(feats)

        featmap_sizes = [featmap.size()[-2:] for featmap in feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        cate_feat, kernel_pred = multi_apply(self.forward_single, feats,
                                            list(range(len(self.seg_num_grids))),
                                            img_metas=img_metas,
                                            eval=eval, upsampled_size=upsampled_size)
        feats_all = []
        for conv, feat in zip(self.lateral_convs, cate_feat):
            feat = conv(F.interpolate(feat, size=(featmap_sizes[0][0],featmap_sizes[0][1]), mode='bilinear', align_corners=True)).unsqueeze(0)
            feats_all.append(feat)
        feats_all = torch.sum(torch.cat(feats_all, dim=0), dim=0)
        feats_all = self.all_conv(feats_all)
        feats_all=self.reasoning(feats_all=feats_all,human_pred=human_pred.detach())
        
        cate_pred, _ = multi_apply(self.forward_single_after, cate_feat,
                                   list(range(len(self.seg_num_grids))),
                                   feats_all=feats_all,
                                   img_metas=img_metas,
                                   eval=eval, upsampled_size=upsampled_size)
        return cate_pred, kernel_pred, human_pred

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], scale_factor=2, mode='bilinear'))

    def forward_single(self, x, idx,  img_metas=None, eval=False, upsampled_size=None):
        ins_kernel_feat = x
        cate_feat=x
        # ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

        # kernel branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear', align_corners=True)


        kernel_feat = kernel_feat.contiguous()
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)

        # cate branch
        cate_feat = cate_feat.contiguous()
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)

        return cate_feat, kernel_pred

    def forward_single_after(self, x, idx, feats_all=None,img_metas=None, eval=False, upsampled_size=None):
        seg_num_grid = self.seg_num_grids[idx]
        
        
        cate_feat = F.interpolate(x, size=seg_num_grid, mode='bilinear', align_corners=True).contiguous()
        feats_all = F.interpolate(feats_all, size=seg_num_grid, mode='bilinear', align_corners=True).contiguous()
        cate_feat=self.sa_list[idx](cate_feat)*feats_all+cate_feat
        for i, cate_layer in enumerate(self.cate_convs_after):
            cate_feat = cate_layer(cate_feat)
            
        cate_pred = self.solo_cate(cate_feat)

        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_pred, None

    def loss(self,
             cate_preds,
             kernel_preds,
             human_pred,
             ins_pred,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             gt_instance_list,
             gt_semantic_seg_list,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        mask_feat_size = ins_pred.size()[-2:]

        human_label_list,human_ind_list=multi_apply(
            self.human_single,
            gt_instance_list,
            gt_bbox_list,
            gt_mask_list,
            mask_feat_size=mask_feat_size
        )
        human_label=torch.cat(human_label_list)
        human_ind=torch.cat(human_ind_list)
        human_label=human_label[human_ind]

        human_pred=human_pred.reshape(-1,mask_feat_size[0]//2,mask_feat_size[1]//2)
        human_pred=human_pred[human_ind]
        # print(img_metas)
        # for i in range(human_label.shape[0]):
        #     plt.imshow(human_label[i].detach().cpu().numpy())
        #     plt.show()

        loss_human=self.loss_human(human_pred,human_label)
        
        semantic_label_list,ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = multi_apply(
            self.solov2_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            gt_semantic_seg_list,
            mask_feat_size=mask_feat_size)
        
        # ins
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]
        # generate masks
        ins_pred = ins_pred
        ins_pred_list = []
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):

                if kernel_pred.size()[-1] == 0:
                    continue
                cur_ins_pred = ins_pred[idx, ...]
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape
                cur_ins_pred = cur_ins_pred.unsqueeze(0)
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(-1, H, W)
                b_mask_pred.append(cur_ins_pred)
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        loss_ins = []
        for input, target in zip(ins_pred_list, ins_labels):
            if input is None:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(self.loss_ins(input, target).unsqueeze(0))
        loss_ins = torch.cat(loss_ins).mean()

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)
        '''
        weight=torch.Tensor([0.9781,0.9557,1.0731,1.0411,0.9562,0.9897,\
            0.963,1.0086,0.9588,0.9588,1.0607,1.0537,0.9556,0.9739,0.9735,\
            1.0328,1.0328,1.0184,1.0186]).expand(flatten_cate_preds.shape).to(flatten_cate_preds.device)'''
        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        
        #loss seg
        # seg_labels = torch.cat([
        #     semantic_label_img.flatten()
        #                for semantic_label_img  in semantic_label_list
        # ])
        # seg_pred =  seg_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        # loss_seg = self.loss_seg(seg_pred,seg_labels,avg_factor=(seg_labels!=self.num_classes).sum()+1)

        return dict(
            #loss_seg=loss_seg,
            loss_ins=loss_ins,
            loss_cate=loss_cate,
            loss_human=loss_human)

    def human_single(self,
                     gt_instance_raw,
                     gt_bboxes_raw,
                     gt_masks_raw,
                     mask_feat_size):
        device = gt_instance_raw[0].device
        human_label = torch.zeros([self.grid_big ** 2, mask_feat_size[0]//2, mask_feat_size[1]//2], dtype=torch.uint8,
                                  device=device)
        human_ind = torch.zeros([self.grid_big ** 2], dtype=torch.bool, device=device)
        human_ids = torch.unique(gt_instance_raw)
        for ids in human_ids:

            keep = (gt_instance_raw == ids)
            gt_instance = gt_instance_raw[keep.cpu().numpy()]
            gt_bboxes = gt_bboxes_raw[keep.cpu().numpy()]
            gt_masks = gt_masks_raw[keep.cpu().numpy()]

            left = torch.min(gt_bboxes[:, 0])
            top = torch.min(gt_bboxes[:, 1])
            right = torch.max(gt_bboxes[:, 2])
            bottom = torch.max(gt_bboxes[:, 3])
            if ((right - left) <= 4) or ((bottom - top) <= 4):
                continue
            gt_masks = gt_masks.to_ndarray()
            gt_masks = np.sum(gt_masks, axis=0).astype(np.uint8)

            half_w = 0.5 * (right - left) * self.sigma
            half_h = 0.5 * (bottom - top) * self.sigma

            upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
            center_h, center_w = ndimage.measurements.center_of_mass(gt_masks)
            coord_w = int((center_w / upsampled_size[1]) // (1. / self.grid_big))
            coord_h = int((center_h / upsampled_size[0]) // (1. / self.grid_big))

            # left, top, right, down
            top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / self.grid_big)))
            down_box = min(self.grid_big - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / self.grid_big)))
            left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / self.grid_big)))
            right_box = min(self.grid_big - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / self.grid_big)))

            top = max(top_box, coord_h - 1)
            down = min(down_box, coord_h + 1)
            left = max(coord_w - 1, left_box)
            right = min(right_box, coord_w + 1)

            output_stride = 8
            gt_masks = mmcv.imrescale(gt_masks, scale=1. / output_stride)
            gt_masks = torch.Tensor(gt_masks)
            for i in range(top, down + 1):
                for j in range(left, right + 1):
                    label = int(i * self.grid_big + j)
                    human_ind[label] = 1
                    human_label[label, :gt_masks.shape[0], :gt_masks.shape[1]] = gt_masks
        return human_label, human_ind

    def solov2_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               gt_semantic_raw,
                               mask_feat_size):

        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.seg_num_grids):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            cate_label = torch.ones([num_grid, num_grid], dtype=torch.int64, device=device)*self.num_classes
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            output_stride = 4
            for seg_mask, gt_label, half_h, half_w in zip(gt_masks, gt_labels, half_hs, half_ws):
                if seg_mask.sum() == 0:
                   continue
                # mass center
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                cate_label[top:(down+1), left:(right+1)] = gt_label
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.Tensor(seg_mask)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            ins_label = torch.stack(ins_label, 0)

            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
            #semantic
        
        gt_semantic_raw = gt_semantic_raw[0].cpu().numpy()
        gt_semantic_raw=cv2.resize(gt_semantic_raw,(mask_feat_size[1],mask_feat_size[0]),interpolation=cv2.INTER_NEAREST)
        gt_semantic_raw_seg=gt_semantic_raw
        gt_semantic_raw_edge=gt_semantic_raw
        #semantic
        
        #gt_semantic_raw_edge=cv2.resize(gt_semantic_raw_edge,(mask_feat_size[1],mask_feat_size[0]),interpolation=cv2.INTER_NEAREST)
        gt_semantic=torch.from_numpy(gt_semantic_raw_seg).to(device).unsqueeze(0).long()
        gt_semantic=(gt_semantic-1)
        gt_semantic[(gt_semantic>self.num_classes)|(gt_semantic<0)]=self.num_classes
        return gt_semantic, ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def get_seg(self, cate_preds, kernel_preds, human_preds, seg_pred, img_metas, cfg, rescale=None):
        num_levels = len(cate_preds)
        featmap_size = seg_pred.size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.num_classes).detach() for i in range(num_levels)
            ]
            seg_pred_list = seg_pred[img_id, ...].unsqueeze(0)
            kernel_pred_list = [
                kernel_preds[i][img_id].permute(1, 2, 0).view(-1, self.kernel_out_channels).detach()
                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            kernel_pred_list = torch.cat(kernel_pred_list, dim=0)

            result = self.get_seg_single(cate_pred_list, seg_pred_list, kernel_pred_list,
                                         featmap_size, img_shape, ori_shape, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list

    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       kernel_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False, debug=False):

        assert len(cate_preds) == len(kernel_preds)

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.)
        inds = (cate_preds > cfg.score_thr)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return [None]*6

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        I, N = kernel_preds.shape
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()
        # mask.
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return [None]*6

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=cfg.kernel, sigma=cfg.sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            return [None]*6
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_shape[:2],
                                  mode='bilinear').squeeze(0)
        seg_masks=seg_masks>cfg.mask_thr
        #semantic
        '''semantic_pred=torch.zeros([self.num_classes,ori_shape[0],ori_shape[1]],device=cate_preds.device)
        semantic_mask=torch.zeros([self.num_classes,ori_shape[0],ori_shape[1]],device=cate_preds.device)
        for i in range(len(cate_labels)):
            label=cate_labels[i]
            mask=seg_masks[i]
            semantic_pred[label,...]=torch.max(semantic_pred[label,...],mask)
        se_max=torch.max(semantic_pred,dim=0)[0]
        se_max[se_max<cfg.mask_thr]=0
        for i in range(semantic_pred.shape[0]):
            semantic_mask[i][(semantic_pred[i]==se_max)&(semantic_pred[i]>0)]=1
        semantic_mask=semantic_mask>cfg.mask_thr'''
        semantic_pred=torch.zeros([self.num_classes,ori_shape[0],ori_shape[1]],dtype=torch.bool, device=cate_preds.device)
        for i in range(len(cate_labels)):
            label=cate_labels[i]
            mask=seg_masks[i]
            
            semantic_pred[label,...]=semantic_pred[label,...]|(mask & (torch.logical_not(torch.sum(semantic_pred,dim=0))))
        
        semantic_mask=semantic_pred     
        sem_labels=torch.linspace(0,self.num_classes-1,self.num_classes,dtype=torch.int8)
        sem_scores=torch.ones([self.num_classes],dtype=torch.uint8)
        return seg_masks, cate_labels, cate_scores, semantic_mask,sem_labels,sem_scores
