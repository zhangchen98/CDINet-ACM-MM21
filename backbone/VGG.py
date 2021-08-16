#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CDINet-master
@File ：VGG.py
@Author ：chen.zhang
@Date ：2021/2/1 9:40
"""

import torch.nn as nn
import torchvision


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 64, 3, 1, 1))
        conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        conv1.add_module('conv1_2', nn.Conv2d(64, 64, 3, 1, 1))
        conv1.add_module('relu1_2', nn.ReLU(inplace=True))
        self.conv1 = conv1
        
        conv2 = nn.Sequential()
        conv2.add_module('pool1', nn.AvgPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(64, 128, 3, 1, 1))
        conv2.add_module('relu2_1', nn.ReLU())
        conv2.add_module('conv2_2', nn.Conv2d(128, 128, 3, 1, 1))
        conv2.add_module('relu2_2', nn.ReLU())
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.AvgPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(128, 256, 3, 1, 1))
        conv3.add_module('relu3_1', nn.ReLU())
        conv3.add_module('conv3_2', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_2', nn.ReLU())
        conv3.add_module('conv3_3', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_3', nn.ReLU())
        self.conv3 = conv3

        conv4 = nn.Sequential()
        conv4.add_module('pool3', nn.AvgPool2d(2, stride=2))
        conv4.add_module('conv4_1', nn.Conv2d(256, 512, 3, 1, 1))
        conv4.add_module('relu4_1', nn.ReLU())
        conv4.add_module('conv4_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv4.add_module('relu4_2', nn.ReLU())
        conv4.add_module('conv4_3', nn.Conv2d(512, 512, 3, 1, 1))
        conv4.add_module('relu4_3', nn.ReLU())
        self.conv4 = conv4

        conv5 = nn.Sequential()
        conv5.add_module('pool4', nn.AvgPool2d(2, stride=2))
        conv5.add_module('conv5_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_1', nn.ReLU())
        conv5.add_module('conv5_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_2', nn.ReLU())
        conv5.add_module('conv5_3', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_3', nn.ReLU())
        self.conv5 = conv5

        vgg_16 = torchvision.models.vgg16(pretrained=True)
        self._initialize_weights(vgg_16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def _initialize_weights(self, vgg_16):
        features = [
            self.conv1.conv1_1, self.conv1.relu1_1,
            self.conv1.conv1_2, self.conv1.relu1_2,
            self.conv2.pool1,
            self.conv2.conv2_1, self.conv2.relu2_1,
            self.conv2.conv2_2, self.conv2.relu2_2,
            self.conv3.pool2,
            self.conv3.conv3_1, self.conv3.relu3_1,
            self.conv3.conv3_2, self.conv3.relu3_2,
            self.conv3.conv3_3, self.conv3.relu3_3,
            self.conv4.pool3,
            self.conv4.conv4_1, self.conv4.relu4_1,
            self.conv4.conv4_2, self.conv4.relu4_2,
            self.conv4.conv4_3, self.conv4.relu4_3,
            self.conv5.pool4,
            self.conv5.conv5_1, self.conv5.relu5_1,
            self.conv5.conv5_2, self.conv5.relu5_2,
            self.conv5.conv5_3, self.conv5.relu5_3,
        ]
        for l1, l2 in zip(vgg_16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
