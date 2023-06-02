import argparse
import os
import shutil
import time
import wandb

import torch
import torch.nn as nn
import torch.nn.parallel

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from wilds import get_dataset
import numpy as np
import random
from datetime import datetime
from cautious_extrapolation.utils import AverageMeter, kl_divergence_gaussian, save_checkpoint, string_to_dict
from cautious_extrapolation.data_paths import DATA_PATHS
import pickle

parser = argparse.ArgumentParser(description='Poverty Eval')
parser.add_argument('--run-name', dest='run_name',
                    type=str, default="")
parser.add_argument('--data-loc', dest='data_loc', default='nfs')

args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "data", args.run_name, "args.txt"), 'r') as f:
    args_train_txt = f.read()
    args_train_dict = string_to_dict(args_train_txt)
checkpoint_name = os.path.join(dir_path, "data", args.run_name, "best.th")

model = torch.nn.DataParallel(models.ResNet18(num_classes=2, num_channels=8))
print("loading checkpoint '{}'".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name)
model.load_state_dict(checkpoint['state_dict'])
model.cuda()

dataset = get_dataset(dataset="poverty", download=True, root_dir=DATA_PATHS[args.data_loc]["poverty"])




def validate(loader, model):
    model.eval()

    outputs_all = np.zeros((len(loader.dataset), 3))

    with torch.no_grad():
        for i, (input, target, metadata) in enumerate(loader):
            target = target.cuda().squeeze()
            input_var = input.cuda()
            output = model(input_var)

            mean = output[:, 0]
            log_var = torch.clip(output[:, 1], -20, 2)
            var = torch.exp(log_var)

            mean = mean.float().cpu().numpy()
            std = torch.sqrt(var).float().cpu().numpy()
            mse = ((mean-target.cpu().numpy())**2)

            outputs_all[i*args_train_dict['batch_size']:(i+1)*args_train_dict['batch_size'], 0] = mean
            outputs_all[i*args_train_dict['batch_size']:(i+1)*args_train_dict['batch_size'], 1] = std
            outputs_all[i*args_train_dict['batch_size']:(i+1)*args_train_dict['batch_size'], 2] = mse
    return outputs_all


results = {}

val_data = dataset.get_subset(
    "id_val")

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=args_train_dict['batch_size'], shuffle=False,
    num_workers=args_train_dict['workers'], pin_memory=True)
outputs_all = validate(val_loader, model)
results["train"] = outputs_all

ood_countries = ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania', 'angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda']

for country in ood_countries:
    print(country)
    country_metadata_idx = dataset._metadata_map["country"].index(country)
    country_idxs = np.where(dataset._metadata_array[:, 2] == country_metadata_idx)
    country_data = torch.utils.data.Subset(dataset, country_idxs[0])
    country_loader = torch.utils.data.DataLoader(
        country_data,
        batch_size=args_train_dict['batch_size'], shuffle=False,
        num_workers=args_train_dict['workers'], pin_memory=True)
    outputs_all = validate(country_loader, model)
    results[country] = outputs_all

save_name = os.path.join(dir_path, "data", args.run_name, 'outputs.pkl')

with open(save_name, 'wb') as f:
    pickle.dump(results, f)