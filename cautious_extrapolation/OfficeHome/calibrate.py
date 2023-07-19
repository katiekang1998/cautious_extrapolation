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
from models import Network
from dataset import get_datasets
import numpy as np
import models
from torch import nn, optim
from torch.nn import functional as F
from cautious_extrapolation.utils import AverageMeter, accuracy, save_checkpoint, get_rew_pred_loss, string_to_dict, ModelWithTemperature
from cautious_extrapolation.data_paths import DATA_PATHS

parser = argparse.ArgumentParser(description='OfficeHome calibration')

parser.add_argument('--run-name', dest='run_name',
                    type=str, default="")
parser.add_argument('--data-loc', dest='data_loc', default='nfs')
args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "data", args.run_name, "args.txt"), 'r') as f:
    args_train_txt = f.read()
    args_train_dict = string_to_dict(args_train_txt)
checkpoint_name = os.path.join(dir_path, "data", args.run_name, "best.th")

assert(args_train_dict["train_type"] == "xent")


model = torch.nn.DataParallel(Network((3, 224, 224), 65))
print("loading checkpoint '{}'".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name)
model.load_state_dict(checkpoint['state_dict'])
model.cuda()

train_dataset, val_dataset, ood_datasets = get_datasets(DATA_PATHS[args.data_loc]["OfficeHome"])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32, shuffle=True,
    num_workers=4, pin_memory=True)


loaders = []
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32, shuffle=True,
    num_workers=4, pin_memory=True)
loaders.append(val_loader)

for ood_dataset in ood_datasets:
    ood_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True)
    loaders.append(ood_loader)


for i in range(len(loaders)):
    loader = loaders[i]
    temp = ModelWithTemperature(model, 65, lr=0.001).set_temperature(loader)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    np.save(os.path.join(dir_path, "data/"+args.run_name+"/temp_"+str(i)+".npy"), temp.detach().cpu().numpy())