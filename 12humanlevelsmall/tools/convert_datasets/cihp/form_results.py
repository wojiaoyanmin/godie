import os
from PIL import Image as PILImage
import numpy as np
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import matplotlib
import json
from collections import defaultdict
matplotlib.use('TkAgg')
import cv2
import time
import mmcv
import pdb
import os.path as osp
import pycocotools.mask as mask_util
import scipy.io as scio
def get_palette1(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    '''
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    '''
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 2] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 0] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3

    return palette

'''
def get_palette(num_cls):
   
    palette=[0,     0,     0,
               128,     0,     0,
                 0,   128,     0,
               128,   128,     0,
                 0,     0 ,  128,
               128  ,   0 ,  128,
                 0  , 128,   128,
               128 ,  128   ,128,
                64 ,    0 ,    0,
               192  ,   0   ,  0,
                64 ,  128 ,    0,
               192 ,  128    , 0,
                64 ,    0,   128
               192 ,    0   128,
                64 ,  128   128,
               192 ,  128   128
                 0 ,   64     0
               128 ,   64     0
                 0  , 192     0
               128   ,192     0
                 0,    64   128
               128 ,   64   128
                 0  , 192   128
               128   ,192   128
                64 ,   64     0
               192,    64     0
                64,   192     0
               192,  192     0
                64 ,   64   128
               192 ,   64   128
                64 ,  192   128
               192 ,  192   128
                 0,     0    64
               128,     0    64
                 0 ,  128    64
               128 ,  128    64
                 0 ,    0   192
               128  ,   0   192
                 0 ,  128   192
               128  , 128   192
                64 ,    0    64
               192 ,    0    64
                64 ,  128    64
               192 ,  128    64
                64,     0   192
               192  ,   0   192
                64,  128   192
               192 ,  128   192
                 0,    64    64
               128  ,  64    64
                 0  , 192    64
               128 ,  192    64
                 0 ,   64   192
               128 ,   64   192
                 0 ,  192   192
               128 ,  192   192
                64 ,   64    64
               192 ,   64    64
                64 ,  192    64
               192 ,  192    64
                64 ,   64   192
               192 ,   64   192
                64 ,  192   192
               192 ,  192   192
                32 ,    0     0
               160 ,    0     0
                32 ,  128     0
               160 ,  128     0
                32 ,    0   128
               160,     0   128
                32,   128   128
               160,   128   128
                96,     0     0
               224,     0     0
                96,   128     0
               224,   128     0
                96,     0   128
               224,     0   128
                96,   128   128
               224 ,  128   128
                32,    64     0
               160,    64     0
                32,   192     0
               160,   192     0
                32,    64   128
               160,    64   128
                32,   192   128
               160 ,  192   128
                96,    64     0
               224,    64     0
                96 ,  192     0
               224,   192     0
                96,    64   128
               224 ,   64   128
                96 ,  192   128
               224,   192   128
                32 ,    0    64
               160,     0    64
                32,   128    64
               160 ,  128    64
                32 ,    0   192
               160 ,    0   192
                32 ,  128   192
               160,   128   192
                96 ,    0    64
               224 ,    0    64
                96 ,  128    64
               224 ,  128    64
                96 ,    0   192
               224,     0   192
                96,   128   192
               224 ,  128   192
                32 ,   64    64
               160,    64    64
                32,   192    64
               160 ,  192    64
                32 ,   64   192
               160 ,   64   192
                32 ,  192   192
               160   192   192
                96    64    64
               224    64    64
                96   192    64
               224   192    64
                96    64   192
               224    64   192
                96   192   192
               224   192   192
                 0    32     0
               128    32     0
                 0   160     0
               128   160     0
                 0    32   128
               128    32   128
                 0   160   128
               128   160   128
                64    32     0
               192    32     0
                64   160     0
               192   160     0
                64    32   128
               192    32   128
                64   160   128
               192   160   128
                 0    96     0
               128    96     0
                 0   224     0
               128   224     0
                 0    96   128
               128    96   128
                 0   224   128
               128   224   128
                64    96     0
               192    96     0
                64   224     0
               192   224     0
                64    96   128
               192    96   128
                64   224   128
               192   224   128
                 0    32    64
               128    32    64
                 0   160    64
               128   160    64
                 0    32   192
               128    32   192
                 0   160   192
               128   160   192
                64    32    64
               192    32    64
                64   160    64
               192   160    64
                64    32   192
               192    32   192
                64   160   192
               192   160   192
                 0    96    64
               128    96    64
                 0   224    64
               128   224    64
                 0    96   192
               128    96   192
                 0   224   192
               128   224   192
                64    96    64
               192    96    64
                64   224    64
               192   224    64
                64    96   192
               192    96   192
                64   224   192
               192   224   192
                32    32     0
               160    32     0
                32   160     0
               160   160     0
                32    32   128
               160    32   128
                32   160   128
               160   160   128
                96    32     0
               224    32     0
                96   160     0
               224   160     0
                96    32   128
               224    32   128
                96   160   128
               224   160   128
                32    96     0
               160    96     0
                32   224     0
               160   224     0
                32    96   128
               160    96   128
                32   224   128
               160   224   128
                96    96     0
               224    96     0
                96   224     0
               224   224     0
                96    96   128
               224    96   128
                96   224   128
               224   224   128
                32    32    64
               160    32    64
                32   160    64
               160   160    64
                32    32   192
               160    32   192
                32   160   192
               160   160   192
                96    32    64
               224    32    64
                96   160    64
               224   160    64
                96    32   192
               224    32   192
                96   160   192
               224   160   192
                32    96    64
               160    96    64
                32   224    64
               160   224    64
                32    96   192
               160    96   192
                32   224   192
               160   224   192
                96    96    64
               224    96    64
                96   224    64
               224   224    64
                96    96   192
               224    96   192
                96   224   192
               224   224   192]
   return palette'''

def get_palette():
    datafile="data/CIHP/pascal_seg_colormap.mat"
    data=(scio.loadmat(datafile)['colormap'].flatten()*255).astype(int).tolist()
    return data


def form_results(train_json,result_json_file, out_dir):
    with open(train_json,'r') as f:
        image=f.read()
        image=json.loads(image)
    print('load train json successfully')
    with open(result_json_file,'r') as g:
        result=g.read()
        result=eval(result)
    print('load result json successfully')
    global_path=osp.join(out_dir,'global_parsing')
    instance_path=osp.join(out_dir,'instance_parsing')
    mmcv.mkdir_or_exist(global_path)
    mmcv.mkdir_or_exist(instance_path)
    num_images=len(image['images'])
    num_instances=len(result)
    #isntance parsin
    split=defaultdict(list)
    
    for i in range(num_instances):
        id_instance=result[i]["image_id"]
        split[id_instance].append(result[i])
    num=0
    for i in range(num_images):

        id=image['images'][i]['id']
        height=image['images'][i]['height']
        width=image['images'][i]['width']
        global_image=np.zeros([height,width],dtype=int)
        instance_image=np.zeros([height,width],dtype=int)
        file_name=osp.basename(image['images'][i]['file_name'])[:-3]+'png'
        txt_filename=osp.join(instance_path,file_name)[:-3]+'txt'
        split_single=split[i]
        assert split_single[0]['image_id']==id  
        for i,anno in enumerate(split_single):
            category=anno['category_id']+1
            i=i+1
            score=anno['score']
            if score<0.1:
                continue            
            counts=mask_util.decode(anno['segmentation'])
            assert counts.shape==instance_image.shape
            inds=counts.nonzero()
            new_inds= np.argwhere(instance_image[inds[0],inds[1]]==0)
            inds=(inds[0][new_inds[:,0]],inds[1][new_inds[:,0]])
            area=inds[0].shape[0]
            if area<11:
                continue
            instance_image[inds[0],inds[1]]=i            
            global_image[inds[0],inds[1]]=category
            with open(txt_filename,'a') as f:
                f.write('%d %f\n'%(category,score*area))

        output_instance = PILImage.fromarray(np.uint8(instance_image))
        palette=get_palette()
        output_instance.putpalette(palette)
        output_instance.save(os.path.join(instance_path,file_name))

        output_global=PILImage.fromarray(np.uint8(global_image))
        output_global.save(os.path.join(global_path,file_name))
        num=num+1
        print("done:",num)
        # plt.show()#显示图像


def main():
    train_json='data/CIHP/annotations/Instance_test.json'
    result_json_file='work_dirs/segm.json'
    #json_file = 'data/CIHP/annotations/Instance_val.json'
    out_dir = 'work_dirs/mp_results/'
    find_image_name='data/CIHP/'
    mmcv.mkdir_or_exist(out_dir)
    form_results(train_json,result_json_file, out_dir)



if __name__ == '__main__':
    main()
