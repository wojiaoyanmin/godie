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





def main():
    '''data=cv2.imread('work_dirs/mp_results/instance_parsing/0000225.png')
    area=data.shape[0]*data.shape[1]
    index=[]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if str(data[i][j]) not in index:
                index.append(str(data[i][j]))
    print(len(index))'''
    with open('work_dirs/mp_results/instance_parsing/0000225.txt') as f:
        data=f.read()
        pdb.set_trace()
        print(data)
if __name__ == '__main__':
    main()
