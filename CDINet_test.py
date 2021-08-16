#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CDINet-master
@File ：CDINet_test.py
@Author ：chen.zhang
@Date ：2021/2/1 11:00
"""

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from model.model_vgg import CDINet
from setting.dataLoader import test_dataset
from setting.options import opt

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

print('load model...')
gpu_num = torch.cuda.device_count()
if gpu_num == 1:
    print("Use Single GPU-", opt.gpu_id)
    model = CDINet()
elif gpu_num > 1:
    print("Use multiple GPUs-", opt.gpu_id)
    model = CDINet()
    model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load('CDINet_cpts/CDINet.pth'), False)

model.cuda()
model.eval()

test_datasets = ['DUT', 'NJU2K', 'NLPR', 'STERE', 'SSD', 'LFSD', 'RGBD135', 'SIP']
dataset_path = opt.test_path
for dataset in test_datasets:
    print("Testing {} ...".format(dataset))
    save_path = 'test_maps/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    depth_root = dataset_path + dataset + '/depth/'
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, depth_root, gt_root, opt.testsize)
    for i in range(test_loader.size):
        image, depth, gt, name = test_loader.load_data()
        name = name.split('/')[-1]
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        pre = model(image, depth)
        pre = F.interpolate(pre, size=gt.shape, mode='bilinear', align_corners=False)
        pre = pre.sigmoid().data.cpu().numpy().squeeze()
        pre = (pre - pre.min()) / (pre.max() - pre.min() + 1e-8)
        cv2.imwrite(save_path + name, pre*255)
    print("Dataset:{} testing completed.".format(dataset))
print("Test Done!")




