import torch.nn.functional as F
import torch
import torch.nn as nn
import pdb
import numpy as np
from ..utils import multi_apply
from . import matrix_nms
from scipy.ndimage import gaussian_filter,maximum_filter
import matplotlib.pyplot as plt
import os.path as osp
def get_seg_single(preds_semantic, preds_center, preds_offset,
                   featmap_size, filename, img_shape, ori_shape, scale_factor, cfg, num_classes,num_ints, rescale=False):
    '''preds_semantic [torch.Size([8,250, 490]),...]level
       preds_center#[torch.Size([1, 250, 490]),...]level
       preds_offset#[torch.Size([2, 250, 490]),...]level'''
    preds_semantic=preds_semantic[:num_classes,:,:]
    device=preds_semantic[0].device
    h, w, _ = img_shape
    upsampled_size_out = (featmap_size[0]*2, featmap_size[1]*2)
    ins_masks,ins_labels,ins_scores=ins(preds_semantic, preds_center, preds_offset,
                   upsampled_size_out, filename, img_shape, ori_shape, scale_factor, cfg, num_classes,num_ints,h, w)  
    
    #semantic
    semantic_masks, semantic_labels, semantic_scores=semantic(preds_semantic, preds_center, preds_offset,
                   upsampled_size_out, filename, img_shape, ori_shape, scale_factor, cfg, num_classes,num_ints, h, w)
    #semantic_label=torch.from_numpy(np.arange(0,num_classes,1)).to(device)
    
    return ins_masks,ins_labels,ins_scores,semantic_masks, semantic_labels, semantic_scores
def semantic(preds_semantic, preds_center, preds_offset,
                   upsampled_size_out, filename, img_shape, ori_shape, scale_factor, cfg, num_classes,num_ints,h, w):
    
    semantic_masks,semantic_labels,semantic_scores=get_semantic(preds_semantic,cfg)
    if semantic_masks is None:
        return None,None,None
    semantic_masks=F.interpolate(semantic_masks.unsqueeze(0),
                                size=upsampled_size_out,
                                mode='bilinear',align_corners=True)[:,:,:h,:w]
    semantic_masks=F.interpolate(semantic_masks,
                                size=ori_shape[:2],
                                mode='bilinear',align_corners=True).squeeze(0)
    semantic_masks = semantic_masks> cfg.foreground_threshold
    return semantic_masks, semantic_labels, semantic_scores
def ins(preds_semantic, preds_center, preds_offset,
                   upsampled_size_out, filename, img_shape, ori_shape, scale_factor, cfg, num_classes,num_ints,h, w):
    
    center_loc_list,center_score_list=find_instance_center(
        preds_center,
        cfg.nms_padding,
        num_ints=num_ints,
        filename=filename,
        threshold=cfg.center_threshold,    
        top_k=cfg.top_k)
    # center_loc:A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x)
    if center_loc_list==[None]*num_ints:
        return None,None,None
    instance_id,_=group_pixels(        
        center_loc_list,
        preds_offset,
        num_ints=num_ints)
    center_score_list_filter=[]
    for center_score in center_score_list:
        if center_score is None:
            continue
        center_score_list_filter.append(center_score)
    center_score=torch.cat(center_score_list_filter)
    ins_masks,ins_labels,ins_scores=get_ins(
        instance_id,
        preds_semantic,#torch.Size([8, 256, 512])
        center_score,
        num_classes=num_classes,
        foreground_threshold=cfg.foreground_threshold)
    # ins_mask  [k h w,  ,  ]level
    # ins_label [k,  ,  , ]level
    # center_score [k, , ,]level
    if ins_masks is None:
        return None,None,None
    ins_masks=F.interpolate(ins_masks.unsqueeze(0),
                                        size=upsampled_size_out,
                                        mode='bilinear',align_corners=True)[:,:,:h,:w]
    ins_masks=F.interpolate(ins_masks,
                                        size=ori_shape[:2],
                                        mode='bilinear',align_corners=True).squeeze(0)
    ins_masks = ins_masks > 0.5

    return ins_masks,ins_labels,ins_scores

def get_semantic(pred_semantic,cfg):
    pred_foreground=torch.zeros_like(pred_semantic)
    pred_foreground[pred_semantic>cfg.foreground_threshold]=pred_semantic[pred_semantic>cfg.foreground_threshold]
    semantic_score=torch.sum(torch.sum(pred_foreground,dim=-1),dim=-1)
    semantic_num=torch.sum(torch.sum((pred_semantic>cfg.foreground_threshold),dim=-1),dim=-1)
    keep=semantic_num>0
    semantic_label=keep.nonzero().flatten()
    semantic_mask=pred_semantic[keep,...]
    semantic_score=semantic_score[keep]/semantic_num[keep]
    if len(semantic_label)==0:
        return None, None, None
    return semantic_mask,semantic_label,semantic_score

def get_ins(instance_id,pred_semantic,center_score,num_classes=None,foreground_threshold=None):
    '''preds_semantic torch.Size([8, 250, 490])
      instance_id#250, 490
      center_loc= [K, 2] '''
    if instance_id==None:
        return None,None,None
    device = pred_semantic.device
    semantic_max=torch.argmax(pred_semantic,dim=0,keepdim=False)#h*w（每个点上的值都是类别,所有的，不管是前景还是背景）
    semantic_label=torch.ones([*(pred_semantic.size()[-2:])],dtype=torch.int64, device=device)*num_classes
    foreground,_=torch.max(pred_semantic>foreground_threshold,dim=0, keepdim=False)#250,490
    semantic_label[foreground]=semantic_max[foreground]
    ids=torch.unique(instance_id)
    ins_num=ids.shape
    ins_mask=torch.zeros([ins_num[0],*(pred_semantic.size()[-2:])],dtype=torch.float,device=device)
    ins_label=torch.ones([ins_num[0]],dtype=torch.bool,device=device)*num_classes
    ins_score=torch.zeros([ins_num[0]],dtype=torch.float,device=device)
    for i in range(ins_num[0]):
        ins_mask_single=(instance_id==ids[i])&(foreground==1)
        if ins_mask_single.sum()==0:
            continue
        class_id,_=torch.mode(semantic_label[ins_mask_single])
        ins_label[i]=class_id
        ins_mask[i,ins_mask_single] = pred_semantic[class_id,ins_mask_single]
        ins_score[i]=ins_mask[i,ins_mask_single].mean()
    if ins_mask.sum()==0:
        return None,None,None
    inds_instance=(ins_label-num_classes).nonzero().squeeze(-1)
    ins_label=ins_label[inds_instance]
    ins_mask=ins_mask[inds_instance,...]
    center_score=center_score[inds_instance]
    ins_score=ins_score[inds_instance]
    score=ins_score*center_score
    if len(ins_label)==0:
        return None,None,None
    return ins_mask,ins_label,score

def find_instance_center(centermap_list,nms_padding,num_ints=None,filename=None, threshold=0.1,top_k=None):
    '''centermap#torch.Size([1, 250, 490])
    return center inds center score'''
    # thresholding, setting values below threshold to -1
    
    filename=osp.split(filename)[-1]
    device=centermap_list[0].device
    '''centermap=centermap.cpu().numpy()
    centermap[0]=gaussian_filter(centermap[0],sigma=0.5)
    centermap=torch.from_numpy(centermap).to(device)'''
    #nms
    
    '''plt.imshow(centermap.cpu().numpy())
    plt.title(filename)
    plt.show()'''
    centermap_list,_=multi_apply(
        find_level_center,
        centermap_list,
        nms_padding,
        threshold=threshold)
    centermap=torch.cat(centermap_list)
    centermap,_=torch.max(centermap,dim=0,keepdim=True)
    centermap,_=find_level_center(centermap,nms_padding[0],threshold=threshold)
    #nonzero elements
    inds_list=[]
    center_score_list=[]
    for center in centermap_list:
        center[center!=centermap]=-1 
    for centermap in centermap_list:
        centermap=centermap.squeeze(0)
        inds=torch.nonzero(centermap>0,as_tuple=False)
        if inds.size(0)==0:
            inds_list.append(None)
            center_score_list.append(None)
        elif top_k is None:
            center_score=centermap[inds[:,0],inds[:,1]]
            inds_list.append(inds)
            center_score_list.append(center_score)        
        elif inds.size(0)<top_k:
            center_score = centermap[inds[:,0], inds[:, 1]]
            inds_list.append(inds)
            center_score_list.append(center_score)
        else:
            #find top k centers
            top_k_scores,_=torch.topk(torch.flatten(centermap[inds[:,0],inds[:,1]]),top_k)
            inds=torch.nonzero(centermap>=top_k_scores[-1],as_tuple=False)
            center_score = centermap[inds[:,0], inds[:, 1]]
            inds_list.append(inds)
            center_score_list.append(center_score)

    return inds_list,center_score_list

def find_level_center(centermap,nms_padding,threshold=None):
    nms_kernel=2*nms_padding+1
    centermap_maxpooled=F.max_pool2d(centermap,kernel_size=nms_kernel,stride=1,padding=nms_padding)
    centermap[centermap!=centermap_maxpooled]=-1 #不是区域极大值的都设置为-1
    #squeeze the fist dimension
    centermap=F.threshold(centermap,threshold,-1)
    return centermap,None

def group_pixels(center_list,preds_offset,num_ints=None):
    """
        Gives each pixel in the image an instance id.
        Arguments:
            center: A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
            offsets: A Tensor of shape [2, H, W] of raw offset output, where N is the batch size,
                for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
        Returns:
            A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    """
    assert len(center_list)==len(preds_offset),'dimension mismatch'
    if center_list==[None,None,None,None]:
        return None,None
    height,width=preds_offset[0].size()[1:]
    # generates a coordinate map, where each location is the coordinate of that loc
    y_coord = torch.arange(height, dtype=preds_offset[0].dtype, device=preds_offset[0].device).repeat(1, width, 1).transpose(1, 2)
    x_coord = torch.arange(width, dtype=preds_offset[0].dtype, device=preds_offset[0].device).repeat(1, height, 1)
    coord = torch.cat((y_coord, x_coord), dim=0)
    offset_list = [coord + offsets for offsets in preds_offset]
    offset_list = [offset.reshape((2, height * width)).transpose(1, 0).unsqueeze(0) for offset in offset_list]#H*W, 2
    # ctr: [K, 2] -> [K, 1, 2]
    # ctr_loc = [H*W, 2] -> [1, H*W, 2]
    # distance: [K, H*W]
    distance=[]
    for center,offset in zip(center_list,offset_list):
        if center is None:
            continue
        distance.append(torch.norm(center.unsqueeze(1) - offset, dim=-1))
    if len(distance)==0:
        return None,None
    distance=torch.cat(distance)
    # 求两点之间的距离[K, H*W]
    # finds center with minimum distance at each location
    instance_id = torch.argmin(distance, dim=0).reshape((height, width))
    return instance_id,None#h*w  0不是instance id
