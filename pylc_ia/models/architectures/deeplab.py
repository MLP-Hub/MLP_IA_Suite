"""
Adapted from Jianfeng Zhang, Vision & Machine Learning Lab, National University of Singapore,
Deeplab V3+ in PyTorch, https://github.com/jfzhang95/pytorch-deeplab-xception

PyTorch implementation of Deeplab: This is a PyTorch(0.4.1) implementation of DeepLab-V3-Plus.
It can use Modified Aligned Xception and ResNet as backbone.
"""

import torch.nn as nn
import torch.nn.functional as F
from models.modules.aspp import build_aspp
from models.backbone import resnet
from models.decoder import build_decoder


class DeepLab(nn.Module):
    def __init__(self, activ_func, normalizer, backbone='resnet', output_stride=16,
                 n_classes=9, in_channels=3, freeze_bn=False, pretrained=True):
        super(DeepLab, self).__init__()

        # Select encoder
        if backbone == 'resnet':
            self.backbone = resnet.ResNet101(output_stride, normalizer, pretrained=pretrained)

        self.aspp = build_aspp(backbone, output_stride, normalizer)
        self.decoder = build_decoder(n_classes, backbone, normalizer)
        self.normalizer = normalizer
        self.in_channels = in_channels
        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

