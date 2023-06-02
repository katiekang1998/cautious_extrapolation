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



parser = argparse.ArgumentParser(description='Poverty Training')
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=1)
parser.add_argument('--data-loc', dest='data_loc', default='nfs')


def poverty_rgb_color_transform(ms_img, transform):
    from wilds.datasets.poverty_dataset import _MEANS_2009_17, _STD_DEVS_2009_17
    poverty_rgb_means = np.array([_MEANS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))
    poverty_rgb_stds = np.array([_STD_DEVS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))

    def unnormalize_rgb_in_poverty_ms_img(ms_img):
        result = ms_img.detach().clone()
        result[:3] = (result[:3] * poverty_rgb_stds) + poverty_rgb_means
        return result

    def normalize_rgb_in_poverty_ms_img(ms_img):
        result = ms_img.detach().clone()
        result[:3] = (result[:3] - poverty_rgb_means) / poverty_rgb_stds
        return ms_img

    color_transform = transforms.Compose([
        transforms.Lambda(lambda ms_img: unnormalize_rgb_in_poverty_ms_img(ms_img)),
        transform,
        transforms.Lambda(lambda ms_img: normalize_rgb_in_poverty_ms_img(ms_img)),
    ])
    # The first three channels of the Poverty MS images are BGR
    # So we shuffle them to the standard RGB to do the ColorJitter
    # Before shuffling them back
    ms_img[:3] = color_transform(ms_img[[2,1,0]])[[2,1,0]] # bgr to rgb to bgr
    return ms_img

def add_poverty_rand_augment_transform():
    def poverty_color_jitter(ms_img):
        return poverty_rgb_color_transform(
            ms_img,
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1))

    def ms_cutout(ms_img):
        def _sample_uniform(a, b):
            return torch.empty(1).uniform_(a, b).item()

        assert ms_img.shape[1] == ms_img.shape[2]
        img_width = ms_img.shape[1]
        cutout_width = _sample_uniform(0, img_width/2)
        cutout_center_x = _sample_uniform(0, img_width)
        cutout_center_y = _sample_uniform(0, img_width)
        x0 = int(max(0, cutout_center_x - cutout_width/2))
        y0 = int(max(0, cutout_center_y - cutout_width/2))
        x1 = int(min(img_width, cutout_center_x + cutout_width/2))
        y1 = int(min(img_width, cutout_center_y + cutout_width/2))

        # Fill with 0 because the data is already normalized to mean zero
        ms_img[:, x0:x1, y0:y1] = 0
        return ms_img

    strong_transform_steps = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=0.1, scale=(0.9, 1.1)),
        transforms.Lambda(lambda ms_img: poverty_color_jitter(ms_img)),
        transforms.Lambda(lambda ms_img: ms_cutout(ms_img)),
    ]

    return transforms.Compose(strong_transform_steps)


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

    
    # transforms = add_poverty_rand_augment_transform()
    train_data = dataset.get_subset(
        "train",
        # transform = transforms,
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

    criterion = nn.GaussianNLLLoss().cuda()


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


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
    log_vars = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input_var, target, metadata) in enumerate(train_loader):
        target = target.squeeze().cuda()
        input_var = input_var.cuda()

        # compute output
        output = model(input_var)

        mean = output[:, 0]
        log_var = torch.clip(output[:, 1], -20, 1)
        log_vars.update(log_var.mean().item(), input_var.size(0))
        var = torch.exp(log_var)

        loss = criterion(mean, target, var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()
        losses.update(loss.item(), input_var.size(0))

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


def validate(val_loader, model, criterion, epoch, name):
    """
    Run evaluation
    """
    losses = AverageMeter()
    log_vars = AverageMeter()
    mses = AverageMeter()


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input_var, target, metadata) in enumerate(val_loader):
            target = target.cuda().squeeze()
            input_var = input_var.cuda()

            # compute output
            output = model(input_var)
            mean = output[:, 0]
            log_var = torch.clip(output[:, 1], -20, 1)
            log_vars.update(log_var.mean().item(), input_var.size(0))
            var = torch.exp(log_var)

            loss = criterion(mean, target, var)
            loss = loss.float()

            # measure accuracy and record loss
            losses.update(loss.item(), input_var.size(0))

            mse = ((mean - target)**2).mean()
            mses.update(mse.item(), input_var.size(0))

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
