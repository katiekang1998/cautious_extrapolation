import pickle

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
from cautious_extrapolation.utils import AverageMeter, save_checkpoint, get_rew_pred_loss, string_to_dict
from cautious_extrapolation.data_paths import DATA_PATHS
from transformers import BertTokenizerFast, DistilBertTokenizerFast, get_linear_schedule_with_warmup
import os
import math
import time
import argparse
import random
from datetime import datetime

parser = argparse.ArgumentParser(description='Amazon Training')
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 20)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='', type=str)
parser.add_argument('--data-loc', dest='data_loc', default='nfs')


def main():
    global args
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic=True

    now = datetime.now()
    if args.save_dir == '':
        save_dir = 'ood_classifier_seed'+str(args.seed)+now.strftime("_%Y_%m_%d_%H_%M_%S")
    else:
        save_dir = args.save_dir+now.strftime("_%Y_%m_%d_%H_%M_%S")


    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Check the save_dir exists or not
    if not os.path.exists(os.path.join(dir_path, "data", save_dir)):
        os.makedirs(os.path.join(dir_path, "data", save_dir))
        #save args
        with open(os.path.join(dir_path, "data", save_dir, 'args.txt'), 'w') as f:
            for arg in vars(args):
                f.write("{}: {}\n".format(arg, getattr(args, arg)))

    print("Loading dataset")
    dataset = get_dataset(dataset="amazon", download=False, root_dir=DATA_PATHS[args.data_loc]["Amazon"])
    print("Dataset loaded")

    transform = models.initialize_bert_transform()
    val_data = dataset.get_subset(
        "id_val",
        transform = transform,)


    ood_data1 = dataset.get_subset(
        "val",
        transform = transform,
    )

    model = models.initialize_bert_based_model(2)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    val_ood_dataset = ValOodDataset(val_data, ood_data1)
    train_val_ood_dataset, val_val_ood_dataset = torch.utils.data.random_split(val_ood_dataset, [0.9, 0.1])
    train_val_ood_loader = torch.utils.data.DataLoader(
        train_val_ood_dataset, batch_size=20, shuffle=True,
        num_workers=1, pin_memory=True, sampler=None)
    val_val_ood_loader = torch.utils.data.DataLoader(
        val_val_ood_dataset, batch_size=20, shuffle=False,
        num_workers=1, pin_memory=True, sampler=None)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay= 0.01)
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=math.ceil(len(train_val_ood_dataset))*args.epochs,
            num_warmup_steps=0)

    for epoch in range(args.epochs):
        train(train_val_ood_loader, model, criterion, optimizer, epoch, scheduler)
        loss = validate(val_val_ood_loader, model, criterion)
        if epoch == 0:
            best_loss = loss
            is_best = True
        else:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict()}, is_best, os.path.join(dir_path, "data", save_dir))

class ValOodDataset(torch.utils.data.Dataset):
    def __init__(self, val_data, ood_data):
        self.val_data = val_data
        self.ood_data = ood_data

        self.subsampled_idxs = np.load("data/ood1_subsampled_idxs.npy")

        self.shuffled_idxs = np.random.permutation(len(val_data))

    def __len__(self):
        return 2*len(self.subsampled_idxs)

    def __getitem__(self, idx):
        if idx < len(self.subsampled_idxs):
            image = self.val_data[self.shuffled_idxs[idx]][0]
            label = 0
        else:
            image = self.ood_data[self.subsampled_idxs[idx - len(self.subsampled_idxs)]][0]
            label = 1

        return image, label
    
def train(train_loader, model, criterion, optimizer, epoch, scheduler):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.cuda().int()
        target = target.cuda()

        output = model(input)
        # print((output.argmax(axis=-1)==target).double().mean())
        loss = criterion(output, target)

        acc1 = (output.argmax(axis=-1) == target).double().mean() #accuracy(output, target, False)
        losses.update(loss.item(), input.size(0))
        acc.update(acc1.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc=acc))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()
            

def validate(val_loader, model, criterion):

    losses = AverageMeter()
    accs = AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda().int()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            acc1 = (output.argmax(axis=-1) == target).double().mean() #accuracy(output, target, False)
            losses.update(loss.item(), input.size(0))
            accs.update(acc1.item(), input.size(0))

    print(' * Acc {acc.avg:.3f}'.format(acc=accs))

    return losses.avg



if __name__ == '__main__':
    main()
