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
from dataset import ImageNet200
import numpy as np
import pickle
from cautious_extrapolation.utils import AverageMeter, accuracy, save_checkpoint, get_rew_pred_loss, string_to_dict
from cautious_extrapolation.data_paths import DATA_PATHS
from torchvision.datasets import ImageFolder
import torchvision.models as models


parser = argparse.ArgumentParser(description='ImageNet Eval')
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


if args_train_dict["train_type"] == 'xent':
    num_classes = 200
elif args_train_dict["train_type"] == 'xent+1':
    num_classes = 201
elif args_train_dict["train_type"] == 'reward_prediction':
    num_classes = 201

model = models.__dict__[args_train_dict['arch']](num_classes=num_classes)
print("loading checkpoint '{}'".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name)
state_dict = {}
for key in checkpoint['state_dict']:
    state_dict[key.replace("module.", "")] = checkpoint['state_dict'][key]
model.load_state_dict(state_dict)
model.cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


valdir = os.path.join(DATA_PATHS["ada"]["ImageNet"], 'val')
val_dataset = ImageNet200(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=256, shuffle=False,
    num_workers=4, pin_memory=True)


imagenet_r_dataset = ImageFolder(DATA_PATHS[args.data_loc]["ImageNet-R"],
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

imagenet_r_loader = torch.utils.data.DataLoader(
    imagenet_r_dataset,
    batch_size=256, shuffle=False,
    num_workers=4, pin_memory=True)


imagenet_s_dataset = ImageNet200(
    DATA_PATHS[args.data_loc]["ImageNet-S"],
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

imagenet_s_loader = torch.utils.data.DataLoader(
    imagenet_s_dataset,
    batch_size=256, shuffle=False,
    num_workers=4, pin_memory=True)

ood_loaders = [imagenet_r_loader, imagenet_s_loader]

def validate(loader, model, temperature):
    model.eval()

    outputs_all = np.zeros((len(loader.dataset), num_classes))

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            print(i)
            target = target.cuda()
            input_var = input.cuda()
            output = model(input_var)
            output = output.float().cpu().numpy()
            if len(temperature) > 0:
                output = output / np.expand_dims(temperature, axis=0)
            outputs_all[i*args_train_dict['batch_size']:(i+1)*args_train_dict['batch_size']] = output
    return outputs_all


results = {}
if args.ts:
    assert(args_train_dict["train_type"] == "xent")
    temperature = np.load(os.path.join(dir_path, "data", args.run_name, "temp.npy"))
else:
    temperature=[]
outputs_all = validate(val_loader, model, temperature)
results[0] = outputs_all

for ood_idx in range(2):
    if args.ts:
        assert(args_train_dict["train_type"] == "xent")
        temperature = np.load(os.path.join(dir_path, "data", args.run_name, "temp.npy"))
    else:
        temperature=[]
    outputs_all = validate(ood_loaders[ood_idx], model, temperature)
    results[ood_idx+1] = outputs_all


if args.ts:
    save_name = os.path.join(dir_path, "data", args.run_name, "outputs_ts.pkl")
else:
    save_name = os.path.join(dir_path, "data", args.run_name, "outputs.pkl")

with open(save_name, 'wb') as f:
    pickle.dump(results, f)