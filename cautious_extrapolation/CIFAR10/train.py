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
import wandb
import numpy as np
import random
from datetime import datetime
from cautious_extrapolation.utils import AverageMeter, accuracy, save_checkpoint, get_rl_loss
from cautious_extrapolation.data_paths import DATA_PATHS


parser = argparse.ArgumentParser(description='CIFAR10 Training')
parser.add_argument('--seed', dest='seed', type=int, default=48)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--corruption-type', dest='corruption_type',
                    help='Type of corruption to add to evaluation images',
                    type=str, default="impulse_noise")
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

    wandb.init(project="CIFAR10", name=save_dir)
    wandb.config.update(args)

    # Check the save_dir exists or not
    if not os.path.exists(os.path.join("data", save_dir)):
        os.makedirs(os.path.join("data", save_dir))
        #save args
        with open(os.path.join("data", save_dir, 'args.txt'), 'w') as f:
            for arg in vars(args):
                f.write("{}: {}\n".format(arg, getattr(args, arg)))


    if args.train_type == 'xent':
        num_classes = 10
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.train_type == 'xent+1':
        num_classes = 11
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.train_type == 'reward_prediction':
        num_classes = 11
        criterion = get_rl_loss(num_classes)
        
    model = torch.nn.DataParallel(models.resnet20(num_classes=num_classes))
    model.cuda()

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=DATA_PATHS[args.data_loc]["CIFAR10"], train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=DATA_PATHS[args.data_loc]["CIFAR10"], train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    ood_loaders = []
    for corruption_level in range(5):
        ood_loader = torch.utils.data.DataLoader(
            CIFAR10C(args.data_loc, args.corruption_type, corruption_level, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        ood_loaders.append(ood_loader)


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150])

    for epoch in range(args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        loss = validate(val_loader, model, criterion, epoch, "val")

        for corruption_level in range(5):
            validate(ood_loaders[corruption_level], model, criterion, epoch, str(args.corruption_type)+str(corruption_level))

        if epoch % args.save_every == 0:
            if epoch == 0:
                best_loss = loss
                is_best = True
            else:
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict()}, is_best, os.path.join("data", save_dir))
            
    is_best = loss < best_loss
    best_loss = min(loss, best_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict() }, is_best, os.path.join("data", save_dir))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):


        target = target.cuda()
        input_var = input.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        acc = accuracy(output.data, target, output.size(1)>10)
        losses.update(loss.item(), input.size(0))
        accs.update(acc.item(), input.size(0))

        # measure elapsed time

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {losses.avg:.4f}\t'
                  'Acc: {accs.avg:.3f}'.format(
                      epoch, i, len(train_loader), losses=losses, accs=accs))

            wandb.log({f'train/loss': float(losses.avg)}, step=epoch*len(train_loader)+i)
            wandb.log({f'train/accuracy': float(accs.avg)}, step=epoch*len(train_loader)+i)


def validate(val_loader, model, criterion, epoch, name):
    """
    Run evaluation
    """
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc = accuracy(output.data, target, output.size(1)>10)
            losses.update(loss.item(), input.size(0))
            accs.update(acc.item(), input.size(0))

            # measure elapsed time

    print('{name} Epoch: [{epoch}]\t'
        'Loss: {losses.avg:.4f}\t'
        'Acc: {accs.avg:.3f}'.format(epoch=epoch,
            name=name, losses=losses, accs=accs))

    wandb.log({f'{name}/loss': float(losses.avg)})
    wandb.log({f'{name}/accuracy': float(accs.avg)})

    return losses.avg


if __name__ == '__main__':
    main()
