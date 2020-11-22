import os
import numpy as np
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import matplotlib
from random import randint
from PIL import Image
from PIL import ImagePalette
matplotlib.use('TkAgg')
import cv2
import time
import mmcv
import pdb
import numpy as np
import glob
import os.path as osp
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

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
    return palette

def indeximage(dataset_dir, out_dir):
    # pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    # json_file='../../datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json'
    # dataset_dir='data/cityscapes/leftImg8bit/val/'
    num=0
    for Image_file in glob.glob(osp.join(dataset_dir, '*.png')):
        filename=osp.basename(Image_file)
        img=Image.open(Image_file)

        num_ins=np.max(np.asarray(img))
        palette=get_palette(255)
        img.putpalette(palette)
        img.save(Image_file)
        num=num+1
        print('done',num)

def main():
    #json_file = 'data/CIHP/annotations/Instance_val.json'
    dataset_dir = 'work_dirs/Instance_ids/'
    out_dir = 'work_dirs/mp_results/instance_parsing/'
    mmcv.mkdir_or_exist(out_dir)
    indeximage(dataset_dir, out_dir)


if __name__ == '__main__':
    main()
