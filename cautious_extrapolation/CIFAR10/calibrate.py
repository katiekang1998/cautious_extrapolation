import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from dataset import CIFAR10C
import numpy as np
import models
from torch import nn, optim
from torch.nn import functional as F
from cautious_extrapolation.utils import AverageMeter, accuracy, save_checkpoint, get_rew_pred_loss, string_to_dict, ModelWithTemperature
from cautious_extrapolation.data_paths import DATA_PATHS

parser = argparse.ArgumentParser(description='CIFAR10 calibration')

parser.add_argument('--run-name', dest='run_name',
                    type=str, default="")
parser.add_argument('--corruption-type', dest='corruption_type',
                    help='Type of corruption to add to evaluation images',
                    type=str, default="")
parser.add_argument('--corruption-level', dest='corruption_level',
                    type=int, default=0)
parser.add_argument('--data-loc', dest='data_loc', default='nfs')
args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "data", args.run_name, "args.txt"), 'r') as f:
    args_train_txt = f.read()
    args_train_dict = string_to_dict(args_train_txt)
checkpoint_name = os.path.join(dir_path, "data", args.run_name, "best.th")

assert(args_train_dict["train_type"] == "xent")

model = torch.nn.DataParallel(models.resnet20(10))
print("loading checkpoint '{}'".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name)
model.load_state_dict(checkpoint['state_dict'])
model.cuda()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


if args.corruption_type == "":
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=DATA_PATHS[args.data_loc]["CIFAR10"], train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args_train_dict['batch_size'], shuffle=True,
        num_workers=args_train_dict['workers'], pin_memory=True)
else:
    loader = torch.utils.data.DataLoader(
        CIFAR10C(args.data_loc, args.corruption_type, args.corruption_level, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args_train_dict['batch_size'], shuffle=True,
        num_workers=args_train_dict['workers'], pin_memory=True)


temp = ModelWithTemperature(model, 10).set_temperature(loader)
dir_path = os.path.dirname(os.path.realpath(__file__))
if args.corruption_type == "":
    np.save(os.path.join(dir_path, "data/"+args.run_name+"/temp.npy"), temp.detach().cpu().numpy())
else:
    np.save(os.path.join(dir_path, "data/"+args.run_name+"/temp_"+args.corruption_type+"_"+str(args.corruption_level)+".npy"), temp.detach().cpu().numpy())

