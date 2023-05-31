import argparse
import os
import shutil
import time
import wandb

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



parser = argparse.ArgumentParser(description='Poverty Training')
parser.add_argument('--seed', dest='seed', type=int, default=48)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
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
        save_dir = str(args.weight_decay)+'_seed'+str(args.seed)+now.strftime("_%Y_%m_%d_%H_%M_%S")
    else:
        save_dir = args.save_dir+now.strftime("_%Y_%m_%d_%H_%M_%S")
    

    wandb.init(project="poverty", name=save_dir)
    wandb.config.update(args)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Check the save_dir exists or not
    if not os.path.exists(os.path.join(dir_path, "data", save_dir)):
        os.makedirs(os.path.join(dir_path, "data", save_dir))
        #save args
        with open(os.path.join(dir_path, "data", save_dir, 'args.txt'), 'w') as f:
            for arg in vars(args):
                f.write("{}: {}\n".format(arg, getattr(args, arg)))
        
    model = torch.nn.DataParallel(models.ResNet18(num_classes=2, num_channels=8))
    model.cuda()

    cudnn.benchmark = True

    dataset = get_dataset(dataset="poverty", download=True, root_dir=DATA_PATHS[args.data_loc]["poverty"])

    # Get the training set
    train_data = dataset.get_subset(
        "train",
    )

    val_data = dataset.get_subset(
        "id_val")


    ood_data1 = dataset.get_subset(
        "val",
    )

    ood_data2 = dataset.get_subset(
        "test",
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    ood_loader1 = torch.utils.data.DataLoader(
        ood_data1,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    ood_loader2 = torch.utils.data.DataLoader(
        ood_data2,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    ood_loaders = [ood_loader1, ood_loader2]

    criterion = nn.GaussianNLLLoss().cuda()


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)


    for epoch in range(args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

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


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target, metadata) in enumerate(train_loader):


        target = target.cuda()
        input = input.cuda()

        # compute output
        output = model(input)

        mean = output[:, 0]
        log_var = torch.clip(output[:, 1], -20, 2)
        var = torch.exp(log_var)

        loss = criterion(mean, target, var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()

        losses.update(loss.item(), input.size(0))

        # measure elapsed time

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {losses.avg:.4f}\t'.format(
                      epoch, i, len(train_loader), losses=losses))

            wandb.log({f'train/loss': float(losses.avg)}, step=epoch*len(train_loader)+i)


def validate(val_loader, model, criterion, epoch, name):
    """
    Run evaluation
    """
    losses = AverageMeter()
    kls = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target, metadata) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            # compute output
            output = model(input)
            mean = output[:, 0]
            log_var = torch.clip(output[:, 1], -20, 2)
            var = torch.exp(log_var)

            loss = criterion(mean, target, var)
            loss = loss.float()

            kl = kl_divergence_gaussian(mean.detach().cpu().numpy(), (var**0.5).detach().cpu().numpy(), np.ones(len(mean))*0.1118, np.ones(len(mean))*0.7990).mean()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            kls.update(kl, input.size(0))

    print('{name} Epoch: [{epoch}]\t'
        'Loss: {losses.avg:.4f}\t'
        'KL: {kls.avg:.3f}'.format(epoch=epoch,
            name=name, losses=losses, kls=kls))

    wandb.log({f'{name}/loss': float(losses.avg)})
    wandb.log({f'{name}/kl': float(kls.avg)})

    return losses.avg


if __name__ == '__main__':
    main()
