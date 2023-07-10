import argparse
import os
import shutil
import time
import wandb
import copy
import torch
import torch.nn as nn
import torch.nn.parallel

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
import numpy as np
import random
from datetime import datetime
from cautious_extrapolation.utils import AverageMeter, kl_divergence_gaussian, save_checkpoint, string_to_dict
from cautious_extrapolation.data_paths import DATA_PATHS
import dataset
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

model = models.GaussNet()
print("loading checkpoint '{}'".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name)
model.load_state_dict(checkpoint['state_dict'])
model.cuda()

cudnn.benchmark = True

val_dataset = dataset.DatasetVal(DATA_PATHS[args.data_loc]["SkinLesionPixels"])
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)

ood_dataset = dataset.DatasetTest(DATA_PATHS[args.data_loc]["SkinLesionPixels"])
ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset, batch_size=32, shuffle=True)
    

def validate(loader, model):
    model.eval()

    outputs_all = np.zeros((len(loader.dataset), 3))

    with torch.no_grad():
        for i, (xs, ys) in enumerate(loader):

            xs = xs.cuda() # (shape: (batch_size, 3, 64, 64))
            ys = ys.cuda() # (shape: (batch_size))
            x_features = model.feature_net(xs) # (shape: (batch_size, hidden_dim))
            means, log_sigma2s = model.head_net(x_features)
            means = means.squeeze(1)
            log_sigma2s = log_sigma2s.squeeze(1)
            sigma2s = torch.exp(log_sigma2s)
            stds = torch.sqrt(sigma2s)

            mse = torch.pow(ys - means, 2)

            outputs_all[i*32:(i+1)*32, 0] = means.cpu().numpy()
            outputs_all[i*32:(i+1)*32, 1] = stds.cpu().numpy()
            outputs_all[i*32:(i+1)*32, 2] = mse.cpu().numpy()
    return outputs_all


results = {}

outputs_all = validate(val_loader, model)

results["train"] = outputs_all


outputs_all = validate(ood_loader, model)

results["ood"] = outputs_all

save_name = os.path.join(dir_path, "data", args.run_name, 'outputs.pkl')

with open(save_name, 'wb') as f:
    pickle.dump(results, f)