import argparse
import glob
import os.path as osp
import pdb
from PIL import Image
import mmcv
import numpy as np
import pycocotools.mask as maskUtils

cihp_id2label={ 0:'Background',
                1:'Hat',
                2:'Hair',
                3:'Glove',
                4:'Sunglasses',
                5:'Upper-clothes',
                6:'Dress',
                7:'Coat',
                8:'Socks',
                9:'Pants',
                10:'tosor-skin',
                11:'Scarf',
                12:'Skirt',
                13:'Face',
                14:'Left-arm',
                15:'Right-arm',
                16:'Left-leg',
                17:'Right-leg',
                18:'Left-shoe',
                19:'Right-shoe' }

def collect_files(Images_dir, Instance_dir ,Category_dir):
    files = []
    suffix='jpg'
    for Image_file in glob.glob(osp.join(Images_dir, '*.jpg')):
        Instance_file = osp.join(Instance_dir, osp.basename(Image_file))[:-len(suffix)]+'png'
        segm_file = osp.join(Category_dir, osp.basename(Image_file))[:-len(suffix)]+'png'
        files.append((Image_file, Instance_file, segm_file))
    assert len(files), f'No images found in {Images_dir}'
    print(f'Loaded {len(files)} images from {Images_dir}')
    return files


def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    Image_file, Instance_file , segm_file= files
    Instance_img = mmcv.imread(Instance_file, 'unchanged')
    ids = np.unique(Instance_img)
    anno_info = []
    for id in ids:
        if id == 0:
            continue
        category_id = int(id)%len(cihp_id2label)-1  # 把background作为最后一个类别
        instance_id= int(id)//len(cihp_id2label)
        iscrowd = 0
        mask = np.asarray(Instance_img == id, dtype=np.uint8, order='F')
        mask_rle = maskUtils.encode(mask[:, :, None])[0]
        area = maskUtils.area(mask_rle)
        # convert to COCO style XYWH format
        bbox = maskUtils.toBbox(mask_rle)
        # for json encoding
        mask_rle['counts'] = mask_rle['counts'].decode()
        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            instance_id=instance_id,
            bbox=bbox.tolist(),
            area=area.tolist(),
            segmentation=mask_rle)
        anno_info.append(anno)
    video_name = osp.basename(osp.dirname(Image_file))
    info = dict(
        # remove img_prefix for filename
        file_name=osp.join(video_name, osp.basename(Image_file)),
        height=Instance_img.shape[0],
        width=Instance_img.shape[1],
        anno_info=anno_info,
        segm_file=osp.basename(segm_file))
    return info

def cvt_annotations(infos, out_json_name):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for info in infos:
        info['id'] = img_id
        anno_infos = info.pop('anno_info')
        out_json['images'].append(info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1

    for label, name in cihp_id2label.items():
        if label==0:
            continue
        cat = dict(id=label-1, name=name)
        out_json['categories'].append(cat)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')
    mmcv.dump(out_json, out_json_name)
    return out_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert cihp annotations to COCO format')
    parser.add_argument('cihp_path', help='cihp data path')
    parser.add_argument('--Images', default='Images', type=str)
    parser.add_argument('--Categoriy-dir', default='Category_ids', type=str)
    parser.add_argument('--Human-dir', default='Human_ids', type=str)
    parser.add_argument('--Instance-dir', default='Instance_ids', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cihp_path = args.cihp_path
    out_dir = args.out_dir if args.out_dir else cihp_path
    mmcv.mkdir_or_exist(out_dir)


    set_name = dict(
        train='Instance_train.json',
        val='Instance_val.json'     
        )

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(
                print_tmpl='It tooks {}s to convert Cityscapes annotation'):
            files = collect_files(
                osp.join(cihp_path, split, args.Images),
                osp.join(cihp_path, split, args.Instance_dir),
                osp.join(cihp_path, split, args.Categoriy_dir))
            infos = collect_annotations(files, nproc=args.nproc)
            cvt_annotations(infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()
