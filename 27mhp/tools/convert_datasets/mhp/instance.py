import argparse
import glob
import os.path as osp
import pdb
from PIL import Image
import mmcv
import numpy as np
import pycocotools.mask as maskUtils

mhp_id2label={  0: 'Background',
                1: 'Cap/hat',
                2: 'Helmet',
                3: 'Face',
                4: 'Hair',
                5: 'Left-arm',
                6: 'Right-arm',
                7: 'Left-hand',
                8: 'Right-hand',
                9: 'Protector',
                10: 'Bikini/bra',
                11: 'Jacket/windbreaker/hoodie ',
                12: 'Tee-shirt',
                13: 'Polo-shirt',
                14: 'Sweater',
                15: 'Singlet',
                16: 'Torso-skin',
                17: 'Pants',
                18: 'Shorts/swim-shorts',
                19: 'Skirt',
                20: 'Stockings',
                21: 'Socks',
                22: 'Left-boot',
                23: 'Right-boot',
                24: 'Left-shoe',
                25: 'Right-shoe',
                26: 'Left-highheel',
                27: 'Right-highheel',
                28: 'Left-sandal',
                29: 'Right-sandal',
                30: 'Left-leg',
                31: 'Right-leg',
                32: 'Left-foot',
                33: 'Right-foot',
                34: 'Coat',
                35: 'Dress',
                36: 'Robe',
                37: 'Jumpsuit',
                38: 'Other-full-body-clothes',
                39: 'Headwear',
                40: 'Backpack',
                41: 'Ball',
                42: 'Bats',
                43: 'Belt',
                44: 'Bottle',
                45: 'Carrybag',
                46: 'Cases',
                47: 'Sunglasses',
                48: 'Eyewear',
                49: 'Glove',
                50: 'Scarf',
                51: 'Umbrella',
                52: 'Wallet/purse',
                53: 'Watch',
                54: 'Wristband',
                55: 'Tie',
                56: 'Other-accessary',
                57: 'Other-upper-body-clothes',
                58: 'Other-lower-body-clothes', }

def collect_files(Images_dir, Instance_dir ):
    files = []
    print("collencting files")
    suffix='.jpg'
    ii=0
    for Image_file in glob.glob(osp.join(Images_dir, '*.jpg')):
        file_single=[]
        file_single.append(Image_file)
        image_index=osp.basename(Image_file)[:-len(suffix)]
        for instance_file in glob.glob(osp.join(Instance_dir, '*.png')):
            instance_name=osp.basename(instance_file)[:-len(suffix)]
            instance_index,totoal_num,human_index=instance_name.split('_')
            if instance_index==image_index:
                file_single.append(instance_file)
        files.append(file_single)
        ii=ii+1
        print('collect ',ii,' image successfully')
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
    Image_file= files[0]
    Instance_files = files[1:]
    anno_info = []
    suffix='.jpg'
    for Instance_file in Instance_files:

        Instance_name=osp.basename(Instance_file)[:-len(suffix)]
        instance_index,totoal_num,human_index=Instance_name.split('_')
        Instance_img = mmcv.imread(Instance_file, 'unchanged')[:,:,2]
        category_ids = np.unique(Instance_img)
        instance_id=int(human_index)-1
        if instance_id<0:
            print(Instance_file)
            print(instance_id<0)
        for category_id in category_ids:

            if category_id==255:
                continue
            if category_id == 0:
                continue
            iscrowd = 0
            
            if category_id<0:
                print(Instance_file)
                print(category_id<0)
            mask = np.asarray(Instance_img == category_id, dtype=np.uint8, order='F')
            mask_rle = maskUtils.encode(mask[:, :, None])[0]
            area = maskUtils.area(mask_rle)
            # convert to COCO style XYWH format
            bbox = maskUtils.toBbox(mask_rle)
            # for json encoding
            mask_rle['counts'] = mask_rle['counts'].decode()
            category_id=int(category_id)-1
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
            segm_file=None)
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

    for label, name in mhp_id2label.items():
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
        description='Convert mhp annotations to COCO format')
    parser.add_argument('mhp_path', help='mhp data path')
    parser.add_argument('--Images', default='images', type=str)
    parser.add_argument('--Categoriy-dir', default='Category_ids', type=str)
    parser.add_argument('--Human-dir', default='Human_ids', type=str)
    parser.add_argument('--Instance-dir', default='parsing_annos', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mhp_path = args.mhp_path
    out_dir = args.out_dir if args.out_dir else mhp_path
    mmcv.mkdir_or_exist(out_dir)


    set_name = dict(
        train='Instance_train.json',
        val='Instance_val.json',
        test='Instance_test.json'
        )

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(
                print_tmpl='It tooks {}s to convert MHP annotation'):
            files = collect_files(
                osp.join(mhp_path, split, args.Images),
                osp.join(mhp_path, split, args.Instance_dir))
            infos = collect_annotations(files, nproc=args.nproc)
            cvt_annotations(infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()
