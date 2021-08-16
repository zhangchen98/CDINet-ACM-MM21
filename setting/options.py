import argparse

parser = argparse.ArgumentParser()
# train set
parser.add_argument('--epoch', type=int, default=120,
                    help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--batchsize', type=int, default=4,
                    help='training batch size')
parser.add_argument('--trainsize', type=int, default=256,
                    help='training image size')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping margin')
parser.add_argument('--model', type=str, default='vgg16',
                    help='backbone')
parser.add_argument('--decay_rate', type=float, default=0.2,
                    help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40,
                    help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None,
                    help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='6',
                    help='gpu')
parser.add_argument('--rgb_root', type=str, default='../../RGBD_dataset/RGBD_for_train/RGB/',
                    help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='../../RGBD_dataset/RGBD_for_train/depth/',
                    help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='../../RGBD_dataset/RGBD_for_train/GT/',
                    help='the training gt images root')
parser.add_argument('--save_path', type=str, default='./CDINet_cpts/', help='the path to save models and logs')
# test set
parser.add_argument('--testsize', type=int, default=256,
                    help='testing image size')
parser.add_argument('--test_path', type=str, default='../../RGBD_dataset/RGBD_for_test/',
                    help='test dataset path')
parser.add_argument('--test_model', type=str, default='CDINet.pth',
                    help='load the model for testing')
opt = parser.parse_args()
