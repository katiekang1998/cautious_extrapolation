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
from transformers import BertTokenizerFast, DistilBertTokenizerFast, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser(description='Amazon Eval')
parser.add_argument('--run-name', dest='run_name',
                    type=str, default="")
parser.add_argument('--data-loc', dest='data_loc', default='nfs')
args = parser.parse_args()


dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoint_name = os.path.join(dir_path, "data", args.run_name, "best.th")

model = models.initialize_bert_based_model(2)

print("loading checkpoint '{}'".format(checkpoint_name))
checkpoint = torch.load(checkpoint_name)
model.load_state_dict(checkpoint['state_dict'])
model.cuda()

dataset = get_dataset(dataset="amazon", download=False, root_dir=DATA_PATHS["nfs"]["Amazon"])

transform = models.initialize_bert_transform()
val_data = dataset.get_subset(
    "id_val",
    transform = transform,)


ood_data1 = dataset.get_subset(
    "val",
    transform = transform,
)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=20, shuffle=False,
    num_workers=1, pin_memory=False)

ood_loader1 = torch.utils.data.DataLoader(
    ood_data1,
    batch_size=20, shuffle=False,
    num_workers=1, pin_memory=False)



def validate(loader, model):
    model.eval()

    outputs_all = np.zeros((len(loader.dataset), 2))

    with torch.no_grad():
        for i, (input, target, _) in enumerate(loader):
            print(i)
            target = target.cuda()
            input_var = input.cuda()
            output = model(input_var)
            output = output.float().cpu().numpy()
            outputs_all[i*20:(i+1)*20] = output

    return outputs_all


outputs = {}
outputs_all = validate(val_loader, model)
outputs[0] = outputs_all

outputs_all = validate(ood_loader1, model)
outputs[1] = outputs_all

save_name = os.path.join(dir_path, "data", args.run_name, "outputs.pkl")

with open(save_name, 'wb') as f:
    pickle.dump(outputs, f)