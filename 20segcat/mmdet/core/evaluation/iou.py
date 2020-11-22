import numpy as np
import pdb
import torch
def iou_eval(pred,label,num_class=None):
    assert pred.shape==label.shape
    metric=SegmentationMetric(num_class)
    pred = pred.flatten()
    label = label.flatten()
    metric.addBatch(pred, label)
    acc = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    return acc,mIoU

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc

    def genConfusionMatrix(self, imgPredict, imgLabel):
    #def genConfusionMatrix(self, pred_label,gt_label):
        # remove classes from unlabeled pixels in gt image and predict
        imgLabel=imgLabel.flatten()
        imgPredict=imgPredict.flatten()
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        '''class_num=20
        index = (gt_label * class_num + pred_label).astype('int32')
        index=index.flatten()
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))
        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                pdb.set_trace()
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]
        return confusion_matrix'''
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)