import os.path as osp
import pickle
import shutil
import tempfile
import time
from PIL import Image as PILImage
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
import numpy as np
from mmdet.core import encode_mask_results, tensor2imgs, get_classes
import pdb
import gc
import pycocotools.mask as mask_util
from scipy import ndimage
import cv2
import pdb
import os


def get_masks(result, num_classes=8):
    for cur_result in result:
        #ins
        encode = [[] for _ in range(num_classes)]
        if cur_result[0] is None:
            return encode
        ins_mask = cur_result[0].cpu().numpy().astype(np.uint8)
        ins_label = cur_result[1].cpu().numpy().astype(np.int)
        ins_score = cur_result[2].cpu().numpy().astype(np.float)
        assert len(ins_mask)==len(ins_label)==len(ins_score),'dimension wrong'
        num_ins = ins_mask.shape[0]
        for idx in range(num_ins):
            cur_mask = ins_mask[idx, ...]
            rle = mask_util.encode(
                np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
            rst = (rle, ins_score[idx])
            encode[ins_label[idx]].append(rst)
        #semantic
        '''semantic_mask=cur_result[3].cpu().numpy().astype(np.uint8)
        semantic_label=cur_result[4].cpu().numpy().astype(np.int)
        semantic_score=cur_result[5].cpu().numpy().astype(np.float)
        for idx in range(semantic_mask.shape[0]):
            cur_mask=semantic_mask[idx,...]
            rle = mask_util.encode(
                np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
            rst = (rle, semantic_score[idx])
            encode_semantic[semantic_label[idx]].append(rst)'''
    return encode

def get_semantic(result,num_classes=8):
    for cur_result in result:
        encode = []
        if cur_result[0] is None:
            return encode
        ins_mask = cur_result[0].cpu().numpy().astype(np.uint8)
        num_ins = ins_mask.shape[0]
        for idx in range(num_ins):
            cur_mask = ins_mask[idx, ...]
            rle = mask_util.encode(
                np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
            encode.append(rle)
    return encode

def vis_seg(data, result, img_norm_cfg, score_thr, save_dir):
    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    assert len(imgs) == len(img_metas)
    class_names = get_classes('CIHP')

    for img, img_meta, cur_result in zip(imgs, img_metas, result):
        if cur_result[0] is None:
            continue
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        seg_label = cur_result[0].cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1].cpu().numpy()
        score = cur_result[2].cpu().numpy()
        vis_inds = score > score_thr
        seg_label = seg_label[vis_inds]
        num_mask = seg_label.shape[0]
        cate_label = cate_label[vis_inds]
        cate_score = score[vis_inds]

        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]  # 从小到大排
        seg_show = img_show.copy()
        for idx in range(num_mask):
            idx = -(idx + 1)
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
                continue
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            cur_mask_bool = cur_mask.astype(np.bool)
            seg_show[cur_mask_bool] = img_show[cur_mask_bool] * 0.3 + color_mask * 0.7
            cur_cate = cate_label[idx]
            cur_score = cate_score[idx]

            label_text = class_names[cur_cate]
            # label_text += '|{:.02f}'.format(cur_score)
            # center
            center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
            vis_pos = (int(center_x), int(center_y))
            cv2.putText(seg_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green


        
            data_id=os.path.split(img_metas[0]['filename'])[-1]
            mmcv.mkdir_or_exist(save_dir)
            '''seg_show = PILImage.fromarray(seg_show)
            seg_show.save(os.path.join(save_dir,data_id))'''
            mmcv.imwrite(seg_show, os.path.join(save_dir,data_id))
            print(cur_score)
            pdb.set_trace()
        


def single_gpu_test(model,
                    data_loader,
                    show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    num_classes = len(dataset.CLASSES)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            seg_result = model(return_loss=False, rescale=True, **data)
        result = get_masks(seg_result, num_classes=num_classes)
        results.append(result)
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

        batch_size = (
            len(data['img_meta']._data)
            if 'img_meta' in data else data['img'][0].size(0))
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, args, cfg, tmpdir=None):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()

    results = []
    dataset = data_loader.dataset
    num_classes = len(dataset.CLASSES)
    colors = [(np.random.random((1, 3)) * 255).tolist()[0] for i in range(num_classes)]

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        result_single=[]
        with torch.no_grad():
            pred_result = model(return_loss=False, rescale=True, **data)
        pred_result_ins=[pred[0:3] for pred in pred_result]
        pred_result_sem=[pred[3:6] for pred in pred_result]

        encoder_ins = get_masks(pred_result_ins, num_classes=num_classes)
        encoder_sem = get_semantic(pred_result_sem,num_classes=num_classes)
        results.append([encoder_ins,encoder_sem ])
        if args.show:
            vis_seg(data, pred_result_ins, cfg.img_norm_cfg,
                    score_thr=args.show_score_thr, save_dir=args.show_dir)
            '''sem_label=torch.linspace(0,num_classes-1,num_classes,dtype=torch.int8)
            sem_score=torch.ones([num_classes],dtype=torch.uint8)
            pred_result_sem=[(pred[-1],sem_label,sem_score) for pred in pred_result]
            assert len(pred_result_sem)==1,"bushi single batch"'''
            sem_dir=args.show_dir
            sem_dir=osp.join(osp.dirname(sem_dir),'semvis')
            '''vis_seg(data,pred_result_sem,cfg.img_norm_cfg,
                    score_thr=args.show_score_thr, save_dir=sem_dir)'''
        torch.cuda.empty_cache()

        if rank == 0:
            batch_size = (
                len(data['img_meta']._data)
                if 'img_meta' in data else data['img'][0].size(0))

            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)
    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results



