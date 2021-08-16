#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CDINet-master
@File ：model_vgg16.py
@Author ：chen.zhang
@Date ：2021/2/1 9:50
"""

import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from backbone.VGG import VGG16
from modules.EM import RDE, DSE
from modules.DDR import DDR


class CDINet(nn.Module):
    """
    The implementation of "Cross-modality Discrepant Interaction Network
    for RGB-D Salient Object Detection".
    """

    def __init__(self) -> None:
        super(CDINet, self).__init__()

        # Two-stream backbone
        self.vgg_r = VGG16()
        self.vgg_d = VGG16()
        # Number of channels per level for vgg16
        self.channels = [64, 128, 256, 512, 512]

        # Enhancement modules in Encoder
        self.rde_layer1 = RDE(channel=self.channels[0])
        self.rde_layer2 = RDE(channel=self.channels[1])
        self.dse_layer3 = DSE(channel=self.channels[2], ratio=4)
        self.dse_layer4 = DSE(channel=self.channels[3], ratio=4)
        self.dse_layer5 = DSE(channel=self.channels[4], ratio=4)

        # transform layer
        self.conv_mid = nn.Conv2d(self.channels[4], self.channels[4], kernel_size=3, padding=1)
        self.relu_mid = nn.ReLU(inplace=True)

        # decoder
        self.ddr = DDR(self.channels)

    def forward(self, image: Tensor, depth: Tensor) -> Tensor:
        """
        Args:
            image: The input of RGB images, three channels.
            depth: The input of Depth images, single channels.

        Returns: The final saliency maps.

        """
        # Copy the depth map as three channels to fit backbone.
        depth = torch.cat((depth, depth, depth), dim=1)
        decoder_list = []

        # encoder layer 1
        conv1_vgg_r = self.vgg_r.conv1(image)
        conv1_vgg_d = self.vgg_d.conv1(depth)

        # encoder layer 2
        conv2_vgg_d_in = self.rde_layer1(conv1_vgg_r, conv1_vgg_d)
        decoder_list.append(conv2_vgg_d_in)
        conv2_vgg_r = self.vgg_r.conv2(conv1_vgg_r)
        conv2_vgg_d = self.vgg_d.conv2(conv2_vgg_d_in)

        # encoder layer 3
        conv3_vgg_d_in = self.rde_layer2(conv2_vgg_r, conv2_vgg_d)
        decoder_list.append(conv3_vgg_d_in)
        conv3_vgg_r = self.vgg_r.conv3(conv2_vgg_r)
        conv3_vgg_d = self.vgg_d.conv3(conv3_vgg_d_in)

        # encoder layer 4
        conv4_vgg_r_in, conv4_vgg_d_in = self.dse_layer3(conv3_vgg_r, conv3_vgg_d)
        decoder_list.append(conv4_vgg_r_in)
        conv4_vgg_r = self.vgg_r.conv4(conv4_vgg_r_in)
        conv4_vgg_d = self.vgg_d.conv4(conv4_vgg_d_in)

        # encoder layer 5
        conv5_vgg_r_in, conv5_vgg_d_in = self.dse_layer4(conv4_vgg_r, conv4_vgg_d)
        decoder_list.append(conv5_vgg_r_in)
        conv5_vgg_r = self.vgg_r.conv5(conv5_vgg_r_in)
        conv5_vgg_d = self.vgg_d.conv5(conv5_vgg_d_in)
        conv5_vgg_r_out, conv5_vgg_d_out = self.dse_layer5(conv5_vgg_r, conv5_vgg_d)

        # transform layer
        mid_feature = self.relu_mid(self.conv_mid(conv5_vgg_r_out))
        decoder_list.append(mid_feature)

        # decoder
        smap = self.ddr(decoder_list)

        return smap


if __name__ == '__main__':
    rgb = torch.randn([1, 3, 256, 256])
    depth = torch.randn([1, 1, 256, 256])
    model = CDINet()
    model(rgb, depth)
