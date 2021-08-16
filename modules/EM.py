#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CDINet-master
@File ：EM.py
@Author ：chen.zhang
@Date ：2021/2/1 9:52
"""

import torch
import torch.nn as nn
from torch import Tensor


class ChannelAttention(nn.Module):
    """
    The implementation of channel attention mechanism.
    """

    def __init__(self, channel: int, ratio: int = 4) -> None:
        """
        Args:
            channel: Number of channels for the input features.
            ratio: The node compression ratio in the full connection layer.
        """
        super(ChannelAttention, self).__init__()
        # global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(True),
            nn.Linear(channel // ratio, channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: The input feature.

        Returns the feature via channel attention.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        out = torch.mul(x, y)

        return out


class SpatialAttention(nn.Module):
    """
    The implementation of spatial attention mechanism.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: The input feature.

        Returns: A weight map of spatial attention, the size is (b, 1, h, w).

        """
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        weight_map = self.sigmoid(x)

        return weight_map


class RDE(nn.Module):
    """
    The implementation of RGB-induced details enhancement module.
    """

    def __init__(self, channel: int) -> None:
        super(RDE, self).__init__()
        self.conv_pool = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        )

        self.conv1 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, input_rgb: Tensor, input_depth: Tensor) -> Tensor:
        rgbd = torch.cat([input_rgb, input_depth], dim=1)
        feature_pool = self.conv_pool(rgbd)

        x, _ = torch.max(input_depth, dim=1, keepdim=True)
        x = self.conv2(self.conv1(x))
        mask = torch.sigmoid(x)

        depth_enhance = feature_pool * mask + input_depth

        return depth_enhance


class DSE(nn.Module):
    """
    The implementation of depth-induced semantic enhancement module.
    """
    def __init__(self, channel: int, ratio: int = 4) -> None:
        super(DSE, self).__init__()
        self.sa1 = SpatialAttention(kernel_size=3)
        self.sa2 = SpatialAttention(kernel_size=3)

        self.ca1 = ChannelAttention(channel, ratio)
        self.ca2 = ChannelAttention(channel, ratio)

    def forward(self, input_rgb, input_depth):
        # attention level
        map_depth = self.sa1(input_depth)
        input_rgb_sa = input_rgb.mul(map_depth) + input_rgb
        input_rgb_sa_ca = self.ca1(input_rgb_sa)
        
        # feature level
        map_depth2 = self.sa2(input_depth)
        input_depth_sa = input_depth.mul(map_depth2) + input_depth
        input_depth_sa_ca = self.ca2(input_depth_sa)

        return input_rgb_sa_ca + input_depth_sa_ca, input_depth_sa_ca
