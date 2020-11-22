import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss
import torch
import pdb

@weighted_loss
def dice_loss(pred,
              target):
    pred = pred.contiguous().view(pred.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()
    a = torch.sum(pred * target, 1)
    b = torch.sum(pred * pred, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    loss = 1-(2 * a) / (b + c) 
    return loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    """DiceLoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0,class_weight=None):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight=class_weight

    def forward(self, pred, target,inds=None,avg_factor=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        if self.class_weight is not None:
            if inds is None:
                weight=self.class_weight.to(pred.device)
            else:
                weight=self.class_weight[inds].to(pred.device)
        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight=None,
            reduction=self.reduction,
            avg_factor=avg_factor)
        return loss
