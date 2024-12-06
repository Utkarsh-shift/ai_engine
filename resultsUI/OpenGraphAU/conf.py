import argparse
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import os
import argparse
# import pprint
import numpy as np
import torch
import random
# import logging
# import shutil
# import yaml


parser = argparse.ArgumentParser(description='PyTorch Training')
# Datasets
parser.add_argument('--dataset', default='hybrid', type=str, choices=['BP4D','DISFA','hybrid'], help="experiment dataset BP4D / DISFA / hybrid Dataset")

# Param
parser.add_argument('-b','--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-lr', '--learning-rate', default=0.00001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-e', '--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer-eps', default=1e-8, type=float)
parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
parser.add_argument('--evaluate', action='store_true', help='evaluation mode')

# Network and Loss
parser.add_argument('--arc', default='swin_transformer_base', type=str, choices=['resnet18', 'resnet50', 'resnet101',
                    'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base'], help="backbone architecture resnet / swin_transformer")
parser.add_argument('--metric', default="dots", type=str, help="metric for graph top-K nearest neighbors selection")
parser.add_argument('--lam', default=0.001, type=float, help="lambda for adjusting loss")

# Device and Seed
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--seed', default=0, type=int, help='seeding for all random operation')

# Experiment
parser.add_argument('--exp-name', default="Test", type=str, help="experiment name for saving checkpoints")
parser.add_argument('--resume', default='', type=str, metavar='path', help='path to latest checkpoint (default: none)')


#demo img input
parser.add_argument('--input', default='', type=str, metavar='path', help='path to img for predicting')
parser.add_argument('--draw_text', action='store_true', help='draw AU predicting results on img')
parser.add_argument('--stage', default=1, type=int, choices=[1, 2], help='model stage')


# ------------------------------


def parser2dict():
    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ------------------------------

def print_conf(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message



def set_env(cfg):
    # set seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    if 'cudnn' in cfg:
        torch.backends.cudnn.benchmark = cfg.cudnn
    else:
        torch.backends.cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids





# check if dir exist, if not create new folder
def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('{} is created'.format(dir_name))



