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
import pickle
from cautious_extrapolation.utils import AverageMeter, accuracy, save_checkpoint, get_rew_pred_loss, string_to_dict
from cautious_extrapolation.data_paths import DATA_PATHS

parser = argparse.ArgumentParser(description='OfficeHome Eval')
parser.add_argument('--run-name', dest='run_name',
                    type=str, default="")
parser.add_argument('--data-loc', dest='data_loc', default='nfs')
parser.add_argument('--ts', type=bool, default=False)

args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "data", args.run_name, "args.txt"), 'r') as f:
    args_train_txt = f.read()
    args_train_dict = string_to_dict(args_train_txt)
checkpoint_name = os.path.join(dir_path, "data", args.run_name, "best.th")

if args_train_dict["train_type"] == "xent":
    output_dim = 65
else:
    output_dim = 66
        
model = torch.nn.DataParallel(Network((3, 224, 224), output_dim))

print("loading checkpoint '{}'".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name)
model.load_state_dict(checkpoint['state_dict'])
model.cuda()

train_dataset, val_dataset, ood_datasets = get_datasets(DATA_PATHS[args.data_loc]["OfficeHome"])


loaders = []
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32, shuffle=False,
    num_workers=args_train_dict['workers'], pin_memory=True)
loaders.append(val_loader)

for ood_dataset in ood_datasets:
    ood_loader = torch.utils.data.DataLoader(
        ood_dataset,
        batch_size=32, shuffle=False,
        num_workers=args_train_dict['workers'], pin_memory=True)
    loaders.append(ood_loader)


def validate(loader, model, temperature):
    model.eval()

    outputs_all = np.zeros((len(loader.dataset), output_dim))

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            target = target.cuda()
            input_var = input.cuda()
            output = model(input_var)
            output = output.float().cpu().numpy()
            if len(temperature) > 0:
                output = output / np.expand_dims(temperature, axis=0)
            outputs_all[i*32:(i+1)*32] = output
    return outputs_all


results = {}
for loader_idx in range(4):
    if args.ts:
        assert(args_train_dict["train_type"] == "xent")
        temperature = np.load(os.path.join(dir_path, "data", args.run_name, "temp_"+str(loader_idx)+".npy"))
    else:
        temperature=[]
    outputs_all = validate(loaders[loader_idx], model, temperature)
    results[loader_idx] = outputs_all


if args.ts:
    save_name = os.path.join(dir_path, "data", args.run_name, "outputs_ts.pkl")
else:
    save_name = os.path.join(dir_path, "data", args.run_name, "outputs.pkl")

with open(save_name, 'wb') as f:
    pickle.dump(results, f)