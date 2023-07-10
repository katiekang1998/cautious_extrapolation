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
from wilds import get_dataset
import numpy as np
import random
from datetime import datetime
from cautious_extrapolation.utils import AverageMeter, kl_divergence_gaussian, save_checkpoint, accuracy, get_rew_pred_loss
from cautious_extrapolation.data_paths import DATA_PATHS
from transformers import BertTokenizerFast, DistilBertTokenizerFast, get_linear_schedule_with_warmup
import math
from torch.nn.utils import clip_grad_norm_


parser = argparse.ArgumentParser(description='Amazon Training')
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=1)
parser.add_argument('--data-loc', dest='data_loc', default='nfs')
parser.add_argument('--train_type', default='xent',
                    choices=['xent', 'xent+1', 'reward_prediction'],
                    help='Train type')



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
        save_dir = args.train_type+'_seed'+str(args.seed)+now.strftime("_%Y_%m_%d_%H_%M_%S")
    else:
        save_dir = args.save_dir+now.strftime("_%Y_%m_%d_%H_%M_%S")
    

    wandb.init(project="Amazon", name=save_dir)
    wandb.config.update(args)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Check the save_dir exists or not
    if not os.path.exists(os.path.join(dir_path, "data", save_dir)):
        os.makedirs(os.path.join(dir_path, "data", save_dir))
        #save args
        with open(os.path.join(dir_path, "data", save_dir, 'args.txt'), 'w') as f:
            for arg in vars(args):
                f.write("{}: {}\n".format(arg, getattr(args, arg)))
        


    dataset = get_dataset(dataset="amazon", download=True, root_dir=DATA_PATHS[args.data_loc]["Amazon"])

    
    transform = models.initialize_bert_transform()
    train_data = dataset.get_subset(
        "train",
        transform = transform,
    )

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
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    
    ood_loader1 = torch.utils.data.DataLoader(
        ood_data1,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    ood_loader2 = torch.utils.data.DataLoader(
        ood_data2,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    
    ood_loaders = [ood_loader1, ood_loader2]


    if args.train_type == 'xent':
        num_classes = train_data.n_classes
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.train_type == 'xent+1':
        num_classes = train_data.n_classes+1
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.train_type == 'reward_prediction':
        num_classes = train_data.n_classes+1
        criterion = get_rew_pred_loss(num_classes)

    model = models.initialize_bert_based_model(num_classes)
    model.cuda()

    cudnn.benchmark = True


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5,
                                weight_decay= 0.01)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=math.ceil(len(train_loader)) *args.epochs,
        num_warmup_steps=0)


    for epoch in range(args.epochs):

        # train for one epoch
        train(train_loader, val_loader, ood_loaders, model, criterion, optimizer, epoch, scheduler)

        # evaluate on validation set
        loss = validate(val_loader, model, criterion, epoch, "val")

        for ood_idx in range(2):
            validate(ood_loaders[ood_idx], model, criterion, epoch, "ood"+str(ood_idx))

        if epoch % args.save_every == 0:
            if epoch == 0:
                best_loss = loss
                is_best = True
            else:
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict()}, is_best, os.path.join(dir_path, "data", save_dir))
            
    is_best = loss < best_loss
    best_loss = min(loss, best_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict() }, is_best, os.path.join(dir_path, "data", save_dir))


def train(train_loader, val_loader, ood_loaders, model, criterion, optimizer, epoch, scheduler):
    """
        Run one train epoch
    """
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input_var, target, _) in enumerate(train_loader):


        target = target.cuda()
        input_var = input_var.cuda()

        # compute output
        output = model(input_var)
        # dist = torch.nn.functional.softmax(output, dim=1)
        # entropy = -torch.sum(dist * torch.log(dist + 1e-8), dim=1)

        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()

        output = output.float()
        loss = loss.float()

        acc = accuracy(output.data, target, output.size(1)>5)
        losses.update(loss.item(), input_var.size(0))
        accs.update(acc.item(), input_var.size(0))

        # measure elapsed time

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {losses.avg:.4f}\t'
                  'Acc: {accs.avg:.3f} \t'.format(
                      epoch, i, len(train_loader), losses=losses, accs=accs))

            wandb.log({f'train/loss': float(losses.avg)}, step=epoch*len(train_loader)+i)
            wandb.log({f'train/accuracy': float(accs.avg)}, step=epoch*len(train_loader)+i)

        if i% 500 == 0 and i>0:
            validate(val_loader, model, criterion, epoch, "val", num_steps=50)
            for ood_idx in range(2):
                validate(ood_loaders[ood_idx], model, criterion, epoch, "ood"+str(ood_idx), num_steps=50)


def validate(val_loader, model, criterion, epoch, name, num_steps = 0):
    """
    Run evaluation
    """
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input_var, target, _) in enumerate(val_loader):
            target = target.cuda()
            input_var = input_var.cuda()

            # compute output
            output = model(input_var)
            # dist = torch.nn.functional.softmax(output, dim=1)
            # entropy = -torch.sum(dist * torch.log(dist + 1e-8), dim=1)
            loss = criterion(output, target)

            output = output.float()
            loss = loss.float()


            # measure accuracy and record loss
            acc = accuracy(output.data, target, output.size(1)>5)
            losses.update(loss.item(), input_var.size(0))
            accs.update(acc.item(), input_var.size(0))

            if num_steps > 0 and i > num_steps:
                break

    print('{name} Epoch: [{epoch}]\t'
        'Loss: {losses.avg:.4f}\t'
        'Acc: {accs.avg:.3f}\t'.format(epoch=epoch,
            name=name, losses=losses, accs=accs))

    wandb.log({f'{name}/loss': float(losses.avg)})
    wandb.log({f'{name}/accuracy': float(accs.avg)})

    return losses.avg


if __name__ == '__main__':
    main()
