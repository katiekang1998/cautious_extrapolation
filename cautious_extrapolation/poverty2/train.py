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
from cautious_extrapolation.utils import AverageMeter, kl_divergence_gaussian, save_checkpoint
from cautious_extrapolation.data_paths import DATA_PATHS
from wilds.common.data_loaders import get_train_loader, get_eval_loader



parser = argparse.ArgumentParser(description='Poverty Training')
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=1)
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
    

    wandb.init(project="poverty2", name=save_dir)
    wandb.config.update(args)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Check the save_dir exists or not
    if not os.path.exists(os.path.join(dir_path, "data", save_dir)):
        os.makedirs(os.path.join(dir_path, "data", save_dir))
        #save args
        with open(os.path.join(dir_path, "data", save_dir, 'args.txt'), 'w') as f:
            for arg in vars(args):
                f.write("{}: {}\n".format(arg, getattr(args, arg)))
        
    model = models.GaussNet()
    model.cuda()

    cudnn.benchmark = True

    dataset = get_dataset(dataset="poverty", download=True, root_dir=DATA_PATHS[args.data_loc]["poverty"])

    
    # transforms = add_poverty_rand_augment_transform()
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Resize((64, 64))
    )

    val_data = dataset.get_subset(
        "id_val",
        transform=transforms.Resize((64, 64))
    )


    ood_data1 = dataset.get_subset(
        "val",
        transform=transforms.Resize((64, 64))
    )

    ood_data2 = dataset.get_subset(
        "test",
        transform=transforms.Resize((64, 64))
    )
    
    train_loader = get_train_loader("standard", train_data, batch_size=32)

    
    val_loader = get_eval_loader("standard", val_data, batch_size=32)

    ood_loader1 = get_eval_loader("standard", ood_data1, batch_size=32)

    ood_loader2 = get_eval_loader("standard", ood_data2, batch_size=32)
    
    
    ood_loaders = [ood_loader1, ood_loader2]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(args.epochs):

        # train for one epoch
        train(train_loader, model, optimizer, epoch)

        # evaluate on validation set
        loss = validate(val_loader, model, epoch, "val")

        for ood_idx in range(2):
            validate(ood_loaders[ood_idx], model, epoch, "ood"+str(ood_idx))

        if epoch % args.save_every == 0:
            if epoch < 20:
                is_best = False
            elif epoch == 20:
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
    log_vars = AverageMeter()

    # switch to train mode
    model.train()

    for i, (xs, ys, metadata) in enumerate(train_loader):

        xs = xs.cuda() # (shape: (batch_size, 8, 64, 64))
        ys = ys.cuda().squeeze(1) # (shape: (batch_size))

        x_features = model.feature_net(xs) # (shape: (batch_size, hidden_dim))
        if (epoch >= 20) and (epoch < 25):
            x_features = x_features.detach()

        means, log_sigma2s = model.head_net(x_features) # (both has shape: (batch_size, 1))
        means = means.view(-1) # (shape: (batch_size))
        log_sigma2s = log_sigma2s.view(-1) # (shape: (batch_size))
        if epoch < 20:
            loss = torch.mean(torch.pow(ys - means, 2))
        else:
            loss = torch.mean(torch.exp(-log_sigma2s)*torch.pow(ys - means, 2) + log_sigma2s)

        optimizer.zero_grad() # (reset gradients)
        loss.backward() # (compute gradients)
        optimizer.step() # (perform optimization step)


        log_vars.update(log_sigma2s.mean().item(), xs.size(0))

        loss = loss.float()
        losses.update(loss.item(), xs.size(0))

        # measure elapsed time

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {losses.avg:.4f}\t'
                  'Log vars: {log_vars.avg:.4f}\t'.format(
                      epoch, i, len(train_loader), losses=losses, log_vars=log_vars))

            wandb.log({f'train/loss': float(losses.avg)}, step=epoch*len(train_loader)+i)
            wandb.log({f'train/log_vars': float(log_vars.avg)}, step=epoch*len(train_loader)+i)
            losses = AverageMeter()
            log_vars = AverageMeter()


def validate(val_loader, model, epoch, name):
    """
    Run evaluation
    """
    losses = AverageMeter()
    log_vars = AverageMeter()
    mses = AverageMeter()


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (xs, ys, metadata) in enumerate(val_loader):
            xs = xs.cuda() # (shape: (batch_size, 8, 64, 64))
            ys = ys.cuda().squeeze(1) # (shape: (batch_size))
            x_features = model.feature_net(xs) # (shape: (batch_size, hidden_dim))
            means, log_sigma2s = model.head_net(x_features)
            means = means.view(-1) # (shape: (batch_size))
            log_sigma2s = log_sigma2s.view(-1) # (shape: (batch_size))
            log_vars.update(log_sigma2s.mean().item(), xs.size(0))

            loss = torch.mean(torch.exp(-log_sigma2s)*torch.pow(ys - means, 2) + log_sigma2s)
            loss = loss.float()

            # measure accuracy and record loss
            losses.update(loss.item(), xs.size(0))

            mse = torch.mean(torch.pow(ys - means, 2))
            mses.update(mse.item(), xs.size(0))

    print('{name} Epoch: [{epoch}]\t'
        'Loss: {losses.avg:.4f}\t'
        'MSE: {mses.avg:.3f}\t'
        'Log vars: {log_vars.avg:.4f}\t'.format(epoch=epoch,
            name=name, losses=losses, mses=mses, log_vars=log_vars))

    wandb.log({f'{name}/loss': float(losses.avg)})
    wandb.log({f'{name}/log_vars': float(log_vars.avg)})
    wandb.log({f'{name}/mse': float(mses.avg)})

    return losses.avg


if __name__ == '__main__':
    main()
