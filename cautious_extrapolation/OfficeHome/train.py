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
from models import Network
from dataset import get_datasets
import wandb
import numpy as np
import random
from datetime import datetime
from cautious_extrapolation.utils import AverageMeter, accuracy, save_checkpoint, get_rew_pred_loss
from cautious_extrapolation.data_paths import DATA_PATHS


parser = argparse.ArgumentParser(description='OfficeHome Training')
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
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

    wandb.init(project="OfficeHome", name=save_dir)
    wandb.config.update(args)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Check the save_dir exists or not
    if not os.path.exists(os.path.join(dir_path, "data", save_dir)):
        os.makedirs(os.path.join(dir_path, "data", save_dir))
        #save args
        with open(os.path.join(dir_path, "data", save_dir, 'args.txt'), 'w') as f:
            for arg in vars(args):
                f.write("{}: {}\n".format(arg, getattr(args, arg)))


    train_dataset, val_dataset, ood_datasets = get_datasets(DATA_PATHS[args.data_loc]["OfficeHome"])

    

    if args.train_type == 'xent':
        num_classes = 65
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.train_type == 'xent+1':
        num_classes = 66
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.train_type == 'reward_prediction':
        num_classes = 66
        criterion = get_rew_pred_loss(num_classes)
        
    model = torch.nn.DataParallel(Network((3, 224, 224), num_classes))
    model.cuda()

    cudnn.benchmark = True


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    ood_loaders = []
    for ood_dataset in ood_datasets:
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=32, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        ood_loaders.append(ood_loader)


    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=5e-05,
            weight_decay=0
        )

    for epoch in range(args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        loss = validate(val_loader, model, criterion, epoch, "val")

        for ood_idx in range(len(ood_loaders)):
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
    accs = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input_var, target) in enumerate(train_loader):


        target = target.cuda()
        input_var = input_var.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        acc = accuracy(output.data, target, output.size(1)>65)
        losses.update(loss.item(), input_var.size(0))
        accs.update(acc.item(), input_var.size(0))

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
        for i, (input_var, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input_var.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc = accuracy(output.data, target, output.size(1)>65)
            losses.update(loss.item(), input_var.size(0))
            accs.update(acc.item(), input_var.size(0))

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
