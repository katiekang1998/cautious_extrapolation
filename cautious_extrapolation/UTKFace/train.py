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
import wandb
import numpy as np
import random
from datetime import datetime
from cautious_extrapolation.utils import AverageMeter, save_checkpoint
from cautious_extrapolation.data_paths import DATA_PATHS
import pandas as pd
from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser(description='UTKFace Training')
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
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
        save_dir = 'seed'+str(args.seed)+now.strftime("_%Y_%m_%d_%H_%M_%S")
    else:
        save_dir = args.save_dir+now.strftime("_%Y_%m_%d_%H_%M_%S")

    wandb.init(project="UTKFace", name=save_dir)
    wandb.config.update(args)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Check the save_dir exists or not
    if not os.path.exists(os.path.join(dir_path, "data", save_dir)):
        os.makedirs(os.path.join(dir_path, "data", save_dir))
        #save args
        with open(os.path.join(dir_path, "data", save_dir, 'args.txt'), 'w') as f:
            for arg in vars(args):
                f.write("{}: {}\n".format(arg, getattr(args, arg)))
        
    model = models.NN()
    model.cuda()

    cudnn.benchmark = True

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49,), (0.23,))
    ])

    dataFrame = pd.read_csv(os.path.join(DATA_PATHS[args.data_loc]["UTKFace"], 'age_gender.gz'), compression='gzip')
    train_dataFrame, val_dataFrame = train_test_split(dataFrame, test_size=0.2, random_state=42)



    train_loader = torch.utils.data.DataLoader(
        UTKDataset(train_dataFrame, transform=normalize),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)
    
    val_loader = torch.utils.data.DataLoader(
        UTKDataset(val_dataFrame, transform=normalize),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    ood_loaders = []
    for corruption_level in range(5):
        ood_loader = torch.utils.data.DataLoader(
            UTKDataset(val_dataFrame, severity=corruption_level+1, transform=normalize),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers)
        ood_loaders.append(ood_loader)


    optimizer =torch.optim.Adam(model.parameters(), lr=args.lr)


    for epoch in range(args.epochs):

        # train for one epoch
        train(train_loader, model, optimizer, epoch)

        # evaluate on validation set
        loss = validate(val_loader, model, epoch, "val")

        for corruption_level in range(5):
            validate(ood_loaders[corruption_level], model, epoch, "ood"+str(corruption_level))

        if epoch <= 5:
            is_best = False
        elif epoch == 6:
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


def train(train_loader, model, optimizer, epoch):
    """
        Run one train epoch
    """
    losses = AverageMeter()
    mses = AverageMeter()
    vars = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input_var, target) in enumerate(train_loader):


        target = target.cuda()
        input_var = input_var.cuda()

        # compute output
        output = model(input_var)
        mean = output[:, 0]
        log_var = output[:, 1]
        var = torch.exp(log_var)


        if epoch <= 5:
            loss = torch.mean((mean - target)**2)
        else:
            loss = torch.mean(torch.exp(-log_var)*torch.pow(target - mean, 2) + log_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        losses.update(loss.item(), input_var.size(0))
        mses.update(torch.mean((mean - target)**2), input_var.size(0))
        vars.update(torch.mean(var), input_var.size(0))

        # measure elapsed time

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {losses.avg:.4f}\t'
                  'MSE: {mses.avg:.3f}\t'
                  'Var: {vars.avg:.3f}\t'.format(
                      epoch, i, len(train_loader), losses=losses, mses=mses, vars=vars))

            wandb.log({f'train/loss': float(losses.avg)}, step=epoch*len(train_loader)+i)
            wandb.log({f'train/mse': float(mses.avg)}, step=epoch*len(train_loader)+i)
            wandb.log({f'train/var': float(vars.avg)}, step=epoch*len(train_loader)+i)
            losses = AverageMeter()
            mses = AverageMeter()


def validate(val_loader, model, epoch, name):
    losses = AverageMeter()
    mses = AverageMeter()
    vars = AverageMeter()

    model.eval()

    for i, (input_var, target) in enumerate(val_loader):


        target = target.cuda()
        input_var = input_var.cuda()

        # compute output
        output = model(input_var)
        mean = output[:, 0]
        log_var = output[:, 1]
        var = torch.exp(log_var)


        if epoch <= 5:
            loss = torch.mean((mean - target)**2)
        else:
            loss = torch.mean(torch.exp(-log_var)*torch.pow(target - mean, 2) + log_var)

        output = output.float()
        loss = loss.float()

        losses.update(loss.item(), input_var.size(0))
        mses.update(torch.mean((mean - target)**2), input_var.size(0))
        vars.update(torch.mean(var), input_var.size(0))


    print('{name} Epoch: [{epoch}]\t'
            'Loss: {losses.avg:.4f}\t'
            'MSE: {mses.avg:.3f}\t'
            'Var: {vars.avg:.3f}\t'.format(
                name=name, epoch=epoch, losses=losses, mses=mses, vars=vars))

    wandb.log({f'{name}/loss': float(losses.avg)}, step=epoch*len(val_loader)+i)
    wandb.log({f'{name}/mse': float(mses.avg)}, step=epoch*len(val_loader)+i)
    wandb.log({f'{name}/var': float(vars.avg)}, step=epoch*len(val_loader)+i)

    return losses.avg


if __name__ == '__main__':
    main()
