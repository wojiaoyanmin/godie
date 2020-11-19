import torch.nn as nn

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import pdb
import torch

@DETECTORS.register_module()
class YMDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 ins_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YMDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        
        if neck is not None:
            self.neck = build_neck(neck)
        # ins_head.update(train_cfg=train_cfg)
        # ins_head.update(test_cfg=test_cfg)
        self.ins_head = build_head(ins_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(YMDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.ins_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
       
        x = self.backbone(img)#[torch.Size([1, 256, 232, 456]),torch.Size([1, 512, 116, 228]).torch.Size([1, 2048, 58, 114])]
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.ins_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_instances=None,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
            gt_semantci_seg:[n,1,h,w]
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)#[1, 256, 258, 506,1, 128, 258, 506]    
        outs = self.ins_head(x,img_metas=img_metas)
        
        loss_inputs = outs + (gt_bboxes, gt_labels,gt_masks,gt_instances,img_metas, self.train_cfg)
        losses = self.ins_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.ins_head(x,img_metas=img_metas, eval=True)
        seg_inputs = outs + (img_metas, self.test_cfg, rescale)

        seg_result = self.ins_head.get_seg(*seg_inputs)
        return seg_result

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError