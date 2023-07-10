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
from dataset import UTKDataset
import numpy as np
import pickle
from cautious_extrapolation.utils import AverageMeter, string_to_dict
from cautious_extrapolation.data_paths import DATA_PATHS
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='UTKFace Eval')
parser.add_argument('--run-name', dest='run_name',
                    type=str, default="")
parser.add_argument('--data-loc', dest='data_loc', default='nfs')

args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "data", args.run_name, "args.txt"), 'r') as f:
    args_train_txt = f.read()
    args_train_dict = string_to_dict(args_train_txt)
checkpoint_name = os.path.join(dir_path, "data", args.run_name, "best.th")


model = models.NN()
print("loading checkpoint '{}'".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name)
model.load_state_dict(checkpoint['state_dict'])
model.cuda()

cudnn.benchmark = True

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49,), (0.23,))
])

dataFrame = pd.read_csv(os.path.join(DATA_PATHS[args.data_loc]["UTKFace"], 'age_gender.gz'), compression='gzip')
train_dataFrame, val_dataFrame = train_test_split(dataFrame, test_size=0.2, random_state=42)

val_loader = torch.utils.data.DataLoader(
    UTKDataset(val_dataFrame, transform=normalize),
    batch_size=64, shuffle=False,
    num_workers=4)

ood_loaders = []
for corruption_level in range(5):
    ood_loader = torch.utils.data.DataLoader(
        UTKDataset(val_dataFrame, severity=corruption_level+1, transform=normalize),
        batch_size=64, shuffle=False,
        num_workers=4)
    ood_loaders.append(ood_loader)


def validate(loader, model):
    model.eval()

    outputs_all = np.zeros((len(loader.dataset), 3))

    with torch.no_grad():
        for i, (xs, ys) in enumerate(loader):

            xs = xs.cuda()
            ys = ys.cuda()
            output = model(xs)
            mean = output[:, 0]
            log_var = output[:, 1]
            var = torch.exp(log_var)
            std = torch.sqrt(var)

            mse = torch.pow(ys - mean, 2)

            outputs_all[i*64:(i+1)*64, 0] = mean.cpu().numpy()
            outputs_all[i*64:(i+1)*64, 1] = std.cpu().numpy()
            outputs_all[i*64:(i+1)*64, 2] = mse.cpu().numpy()
    return outputs_all

results = {}
outputs_all = validate(val_loader, model)
results[0] = outputs_all

for corruption_level in range(5):
    outputs_all = validate(ood_loaders[corruption_level], model)
    results[corruption_level+1] = outputs_all


save_name = os.path.join(dir_path, "data", args.run_name, "outputs.pkl")

with open(save_name, 'wb') as f:
    pickle.dump(results, f)