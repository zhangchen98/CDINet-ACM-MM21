#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CDINet-master
@File ：dataloader.py
@Author ：chen.zhang
@Date ：2021/2/1 10:00
"""

import os
import random
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
from torch.utils import data


class SalObjDataset(data.Dataset):

    def __init__(self, image_root, depth_root, gt_root, trainsize):
        """
        :param image_root: The path of RGB training images.
        :param depth_root: The path of depth training images.
        :param gt_root: The path of training ground truth.
        :param trainsize: The size of training images.
        """
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, ], [0.229, ])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        depth = self.binary_loader(self.depths[index])
        gt = self.binary_loader(self.gts[index])
        image, depth, gt = randomFlip(image, depth, gt)
        image, depth, gt = randomRotation(image, depth, gt)
        image = self.img_transform(image)
        depth = self.depths_transform(depth)
        gt = self.gt_transform(gt)
        return image, depth, gt

    def __len__(self):
        return self.size

    def filter_files(self):
        """ Check whether a set of images match in size. """
        assert len(self.images) == len(self.depths) == len(self.gts)
        images = []
        depths = []
        gts = []
        for img_path, depth_path, gt_path in zip(self.images, self.depths, self.gts):
            # Notes: On DUT dataset, the size of training depth images are [256, 256],
            # it is not matched with RGB images and GT [600, 400].
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                depths.append(depth_path)
                gts.append(gt_path)
            else:
                raise Exception("Image sizes do not match, please check.")
        self.images = images
        self.depths = depths
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            # Removing alpha channel.
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')


def randomFlip(img, depth, gt):
    flip_flag = random.randint(0, 2)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_flag == 2:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
        gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
    return img, depth, gt


def randomRotation(image, depth, gt):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        gt = gt.rotate(random_angle, mode)
    return image, depth, gt


def get_loader(image_root, depth_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = SalObjDataset(image_root, depth_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:

    def __init__(self, image_root, depth_root, gt_root, testsize):
        """
        :param image_root: The path of RGB testing images.
        :param depth_root: The path of depth testing images.
        :param gt_root: The path of testing ground truth.
        :param testsize: The size of testing images.
        """
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.depth = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')
                      or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depth = sorted(self.depth)
        self.gts = sorted(self.gts)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, ], [0.229, ])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def __len__(self):
        return self.size

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)
        depth = self.binary_loader(self.depth[self.index])
        depth = self.depth_transform(depth).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, depth, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')


