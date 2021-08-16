#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CDINet-master
@File ：CDINet_train.py
@Author ：chen.zhang
@Date ：2021/2/1 10:00
"""

import os
import time
import random
import logging
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from setting.dataLoader import get_loader, test_dataset
from setting.utils import clip_gradient, adjust_lr
from setting.options import opt
from model.model_vgg import CDINet

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# fixed random seed
def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_torch()

save_path = opt.save_path
if not os.path.exists(save_path):
    print("Create save_path directory...")
    os.mkdir(save_path)

# load data
print('load data...')
train_loader = get_loader(opt.rgb_root, opt.depth_root, opt.gt_root,
                          batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO,
                    filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("CDINet-Train")
logging.info('Config--epoch:{};lr:{};batchsize:{};imagesize:{};decay_epoch:{}'.
             format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.decay_epoch))
writer = SummaryWriter(save_path + 'summary')

# model
print('load model...')
gpu_num = torch.cuda.device_count()
if gpu_num == 1:
    print("Use Single GPU -", opt.gpu_id)
    model = CDINet()
elif gpu_num > 1:
    print("Use multiple GPUs -", opt.gpu_id)
    model = CDINet()
    model = torch.nn.DataParallel(model)

# Restore training from checkpoints
if opt.load is not None:
    model.load_state_dict(torch.load(opt.load))
    print('load model from', opt.load)

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
# optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.9, weight_decay=5e-4)
CE = torch.nn.BCEWithLogitsLoss()


# check model size
# if not os.path.exists('module_size'):
#     os.makedirs('module_size')
# for name, module in model.named_children():
#     torch.save(module, 'module_size/' + '%s' % name + '.pth')


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, depths, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            depths = depths.cuda()
            gts = gts.cuda()
            s = model(images, depths)
            loss = CE(s, gts)
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            epoch_step += 1
            loss_all += loss.detach()
            if i % 50 == 0:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i,
                             total_step, loss_all.data / epoch_step))
        loss_all /= epoch_step
        print('Epoch [{:03d}/{:03d}]:Loss_AVG={:.4f}'.format(epoch, opt.epoch, loss_all))
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'
                     .format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch > 80 and (epoch % 5 == 0 or epoch == opt.epoch):
            torch.save(model.state_dict(), save_path + 'CDINet_epoch_{}.pth'.format(epoch))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        torch.save(model.state_dict(), save_path + 'CDINet_epoch_{}_checkpoint.pth'.format(epoch + 1))
        print('save checkpoint successfully!')
        raise


if __name__ == '__main__':
    print("-------------------Config-------------------")
    print('epoch:\t\t{}\n'
          'lr:\t\t{}\n'
          'batchsize:\t{}\n'
          'image_size:\t{}\n'
          'decay_epoch:\t{}\n'
          'decay_rate:\t{}\n'
          'checkpoint:\t{}'
          .format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.decay_epoch, opt.decay_rate, opt.load))
    print("--------------------------------------------")
    print("Start train...")
    time_begin = time.time()
    # warm_up_with_multistep_lr = lambda epoch: epoch / 5 if epoch <= 5 else \
    #     0.2 ** len([m for m in [40, 80, 100] if m <= epoch])
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
    for epoch in range(1, opt.epoch + 1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # cur_lr = scheduler.get_lr()
        writer.add_scalar('learning-rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        # scheduler.step()
        time_epoch = time.time()
        print("Time out:{:2f}s\n".format(time_epoch - time_begin))
