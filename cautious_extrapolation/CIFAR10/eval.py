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
import pickle
from cautious_extrapolation.utils import AverageMeter, accuracy, save_checkpoint, get_rew_pred_loss, string_to_dict
from cautious_extrapolation.data_paths import DATA_PATHS

parser = argparse.ArgumentParser(description='CIFAR10 Eval')
parser.add_argument('--run-name', dest='run_name',
                    type=str, default="")
parser.add_argument('--corruption-type', dest='corruption_type',
                    help='Type of corruption to add to evaluation images',
                    type=str, default="impulse_noise")
parser.add_argument('--data-loc', dest='data_loc', default='nfs')
parser.add_argument('--ts', type=bool, default=False)

args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "data", args.run_name, "args.txt"), 'r') as f:
    args_train_txt = f.read()
    args_train_dict = string_to_dict(args_train_txt)
checkpoint_name = os.path.join(dir_path, "data", args.run_name, "best.th")

if args_train_dict["train_type"] == "xent":
    output_dim = 10
else:
    output_dim = 11

model = torch.nn.DataParallel(models.resnet20(output_dim))
print("loading checkpoint '{}'".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name)
model.load_state_dict(checkpoint['state_dict'])
model.cuda()



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root=DATA_PATHS[args.data_loc]["CIFAR10"], train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args_train_dict['batch_size'], shuffle=False,
    num_workers=args_train_dict['workers'], pin_memory=True)

ood_loaders = []
for corruption_level in range(5):
    ood_loader = torch.utils.data.DataLoader(
        CIFAR10C(args.data_loc, args.corruption_type, corruption_level, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args_train_dict['batch_size'], shuffle=False,
        num_workers=args_train_dict['workers'], pin_memory=True)
    ood_loaders.append(ood_loader)


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

for corruption_level in range(5):
    if args.ts:
        assert(args_train_dict["train_type"] == "xent")
        temperature = np.load(os.path.join(dir_path, "data", args.run_name, "temp_"+args.corruption_type+"_"+str(corruption_level)+".npy"))
    else:
        temperature=[]
    outputs_all = validate(ood_loaders[corruption_level], model, temperature)
    results[corruption_level+1] = outputs_all


if args.ts:
    save_name = os.path.join(dir_path, "data", args.run_name, "outputs_"+args.corruption_type+'_ts.pkl')
else:
    save_name = os.path.join(dir_path, "data", args.run_name, "outputs_"+args.corruption_type+'.pkl')

with open(save_name, 'wb') as f:
    pickle.dump(results, f)