#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CDINet-master
@File ：DDR.py
@Author ：chen.zhang
@Date ：2021/2/1 9:55
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List


class BaseConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, bias=True) -> None:
        super(BaseConv2d, self).__init__()
        self.basicconv = nn.Sequential(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.basicconv(x)


class DDR(nn.Module):
    """
    The implementation of Dense Decoding Reconstruction (DDR) structure.
    """

    def __init__(self, channels: List[int]) -> None:
        """
        Args:
            channels: It should a list which denotes the same channels
                      of encoder side outputs(skip connection features).
        """
        super(DDR, self).__init__()

        # decoder layer 5
        self.conv5 = nn.Sequential(
            BaseConv2d(channels[4], channels[4]),
            BaseConv2d(channels[4], channels[4]),
            BaseConv2d(channels[4], channels[3]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # decoder layer 4
        self.conv4 = nn.Sequential(
            BaseConv2d(channels[3] * 2, channels[3]),
            BaseConv2d(channels[3], channels[3]),
            BaseConv2d(channels[3], channels[2]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # decoder layer 3
        self.conv3 = nn.Sequential(
            BaseConv2d(channels[2] * 2, channels[2]),
            BaseConv2d(channels[2], channels[2]),
            BaseConv2d(channels[2], channels[1]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # decoder layer 2
        self.conv2 = nn.Sequential(
            BaseConv2d(channels[1] * 2, channels[1]),
            BaseConv2d(channels[1], channels[0]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # decoder layer 1
        self.conv1 = nn.Sequential(
            BaseConv2d(channels[0] * 2, channels[0]),
            BaseConv2d(channels[0], 3)
        )

        self.c1 = nn.Sequential(
            BaseConv2d(channels[4], channels[3], kernel_size=1, padding=0),
            nn.Conv2d(channels[3], channels[3], kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            BaseConv2d(channels[4] + channels[3], channels[2], kernel_size=1, padding=0),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1)
        )

        self.c3 = nn.Sequential(
            BaseConv2d(channels[4] + channels[3] + channels[2], channels[1],
                       kernel_size=1, padding=0),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1)
        )

        self.c4 = nn.Sequential(
            BaseConv2d(channels[4] + channels[3] + channels[2] + channels[1], channels[0],
                       kernel_size=1, padding=0),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1)
        )

        # decoder out
        self.conv_map = nn.Conv2d(3, 1, kernel_size=3, padding=1)
    #     self._initialize_weights()
    #
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv2d, nn.Linear)):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0.0)

    def forward(self, decoder_list: List[Tensor]) -> Tensor:
        assert len(decoder_list) == 5

        # decoder layer 5
        decoder_map5 = self.conv5(decoder_list[4])  # [b, 512, 32, 32]

        # decoder layer 4
        semantic_block4 = self.c1(F.interpolate(decoder_list[4], scale_factor=2))
        assert semantic_block4.size() == decoder_list[3].size()
        short4 = torch.mul(semantic_block4, decoder_list[3]) + decoder_list[3]
        decoder_map4_input = torch.cat([decoder_map5, short4], dim=1)
        decoder_map4 = self.conv4(decoder_map4_input)  # [b, 256, 64, 64]

        # decoder layer 3
        semantic_block3 = self.c2(
            torch.cat([F.interpolate(decoder_list[4], scale_factor=4),
                       F.interpolate(decoder_list[3], scale_factor=2)], dim=1))
        assert semantic_block3.size() == decoder_list[2].size()
        short3 = torch.mul(semantic_block3, decoder_list[2]) + decoder_list[2]
        decoder_map3_input = torch.cat([decoder_map4, short3], dim=1)
        decoder_map3 = self.conv3(decoder_map3_input)  # [b, 128, 128, 128]

        # decoder layer 2
        semantic_block2 = self.c3(
            torch.cat([F.interpolate(decoder_list[4], scale_factor=8),
                       F.interpolate(decoder_list[3], scale_factor=4),
                       F.interpolate(decoder_list[2], scale_factor=2)], dim=1))
        assert semantic_block2.size() == decoder_list[1].size()
        short2 = torch.mul(semantic_block2, decoder_list[1]) + decoder_list[1]
        decoder_map2_input = torch.cat([decoder_map3, short2], dim=1)
        decoder_map2 = self.conv2(decoder_map2_input)  # [b, 64, 256, 256]

        # decoder layer 1
        semantic_block1 = self.c4(
            torch.cat([F.interpolate(decoder_list[4], scale_factor=16),
                       F.interpolate(decoder_list[3], scale_factor=8),
                       F.interpolate(decoder_list[2], scale_factor=4),
                       F.interpolate(decoder_list[1], scale_factor=2)], dim=1))
        assert semantic_block1.size() == decoder_list[0].size()
        short1 = torch.mul(semantic_block1, decoder_list[0]) + decoder_list[0]
        decoder_map1_input = torch.cat([decoder_map2, short1], dim=1)
        decoder_map1 = self.conv1(decoder_map1_input)  # [b, 3, 256, 256]

        # decoder out
        smap = self.conv_map(decoder_map1)  # [b, 1, 256, 256]

        return smap
