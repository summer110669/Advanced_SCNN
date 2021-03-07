import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(backbone, self).__init__()
        self.backbone = models.vgg16_bn(pretrained).features

        # ----------------- process backbone -----------------
        for i in [34, 37, 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
        self.backbone._modules.pop('33')
        self.backbone._modules.pop('43')

    def forward(self, img):
        return self.backbone(img)
