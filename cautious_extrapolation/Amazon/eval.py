import argparse
import os
import shutil

import time

# import IPython; IPython.embed()

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
import pickle
from cautious_extrapolation.utils import AverageMeter, accuracy, save_checkpoint, get_rew_pred_loss, string_to_dict
from cautious_extrapolation.data_paths import DATA_PATHS

parser = argparse.ArgumentParser(description='Amazon Eval')
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

dataset = get_dataset(dataset="amazon", download=False, root_dir=DATA_PATHS[args.data_loc]["Amazon"])

transform = models.initialize_bert_transform()
val_data = dataset.get_subset(
    "id_val",
    transform = transform,)


ood_data1 = dataset.get_subset(
    "val",
    transform = transform,
)

ood_data2 = dataset.get_subset(
    "test",
    transform = transform,
)


val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=args_train_dict['batch_size'], shuffle=False,
    num_workers=1, pin_memory=False)

ood_loader1 = torch.utils.data.DataLoader(
    ood_data1,
    batch_size=args_train_dict['batch_size'], shuffle=False,
    num_workers=1, pin_memory=False)

ood_loader2 = torch.utils.data.DataLoader(
    ood_data2,
    batch_size=args_train_dict['batch_size'], shuffle=False,
    num_workers=1, pin_memory=False)

ood_loaders = [ood_loader1, ood_loader2]


if args_train_dict["train_type"] == "xent":
    output_dim = val_data.n_classes
else:
    output_dim = val_data.n_classes+1

print("here")
model = models.initialize_bert_based_model(output_dim)

print("loading checkpoint '{}'".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name)
model.load_state_dict(checkpoint['state_dict'])
model.cuda()



def validate(loader, model, temperature):
    model.eval()

    outputs_all = np.zeros((len(loader.dataset), output_dim))

    with torch.no_grad():
        for i, (input, target, _) in enumerate(loader):
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
        temperature = np.load(os.path.join(dir_path, "data", args.run_name, "temp_"+args.corruption_type+"_"+str(ood_idx)+".npy"))
    else:
        temperature=[]
    outputs_all = validate(ood_loaders[ood_idx], model, temperature)
    results[ood_idx+1] = outputs_all


if args.ts:
    save_name = os.path.join(dir_path, "data", args.run_name, "outputs_p.pkl")
else:
    save_name = os.path.join(dir_path, "data", args.run_name, "outputs.pkl")

with open(save_name, 'wb') as f:
    pickle.dump(results, f)