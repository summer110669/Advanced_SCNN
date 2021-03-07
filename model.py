import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from backbone import *
from spatial_conv import *

class SCNN(nn.Module):
    def __init__(
            self,
            input_size,
            ms_ks=9,
            pretrained=True
    ):
        """
        Argument
            ms_ks: kernel size in message passing conv
        """
        super(SCNN, self).__init__()
        self.pretrained = pretrained
        self.backbone = backbone(self.pretrained)
        self.spatial_conv = spatial_conv(input_size, ms_ks)
        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1, 1, 1, 1]))
        self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None, exist_gt=None):
        x = self.backbone(img)
        exist_pred = self.spatial_conv(x)

        if seg_gt is not None and exist_gt is not None:
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss_exist = self.bce_loss(exist_pred, exist_gt)
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        else:
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred, exist_pred, loss_seg, loss_exist, loss

    

