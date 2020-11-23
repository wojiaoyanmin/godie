import os
import glob
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import cv2
import time
import pdb
import mmcv


def jsontopicture(json_file, dataset_dir, out_dir):
    # pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    # json_file='../../datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json'
    # dataset_dir='data/cityscapes/leftImg8bit/val/'
    coco = COCO(json_file)
    # catIds=coco.getCatIds(catNms=['person'])#catIds=1表示人这一类
    imgIds = coco.getImgIds()  # 图片id，许多值

    for i in range(len(imgIds)):
        img = coco.loadImgs(imgIds[i])[0]
        plt.clf()
        I = plt.imread(dataset_dir + img['file_name'])

        plt.axis('off')
        plt.imshow(I)  # 绘制图像，显示交给plt.show()处理

        # plt.show()


        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)
        subfile = os.path.split(img['file_name'])
        path = os.path.join(out_dir, '{}'.format(subfile[0]))
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(path)
        plt.savefig(os.path.join(path, '{}'.format(subfile[1])))
        time.sleep(0.5)
        print("done")
        # plt.show()#显示图像


def main():
    json_file='data/MHP/annotations/Instance_val.json'
    #json_file = 'data/CIHP/annotations/Instance_val.json'
    dataset_dir = 'data/MHP/val/'
    out_dir = 'work_dirs/gtjsonvis/'
    mmcv.mkdir_or_exist(out_dir)
    jsontopicture(json_file, dataset_dir, out_dir)


if __name__ == '__main__':
    main()
