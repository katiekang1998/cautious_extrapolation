import torch
import os
import torch.nn as nn
from torch import nn, optim
import numpy as np
import torchvision.models


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, exclude_last):
    if exclude_last:
        return (output[:, :-1].argmax(axis=-1) == target).double().mean()
    else:
        return (output.argmax(axis=-1) == target).double().mean()


def save_checkpoint(state, is_best, filename):
    """
    Save the training model
    """
    torch.save(state, os.path.join(filename,  'checkpoint.th'))
    if is_best:
        torch.save(state, os.path.join(filename,  'best.th'))



def get_rew_pred_loss(num_actions, misspecification_cost=4):
    def rew_pred_loss(output, target_var):
        one_hot = nn.functional.one_hot(target_var.to(torch.int64), num_actions)
        reward = (misspecification_cost+1)*one_hot - misspecification_cost
        reward[:, -1] = 0
        loss = torch.mean((output-reward)**2)
        return loss
    return rew_pred_loss


def string_to_dict(s):
    # Split the string into lines
    lines = s.split('\n')

    # Initialize an empty dictionary
    d = {}

    # For each line in the lines
    for line in lines:
        # Split the line into key and value on ': '
        split_line = line.split(': ')

        # If there are not 2 elements after split, skip this line
        if len(split_line) != 2:
            continue

        key, value = split_line

        # Try to convert value to integer, if fails, then to float, if fails, keep as string
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass

        # Add key-value pair to dictionary
        d[key] = value

    return d



class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, num_classes, lr = 0.01, max_iter = 1000):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.max_iter = max_iter
        self.temperature = nn.Parameter(torch.ones(num_classes) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), self.num_classes)
        return logits / temperature

    def set_temperature(self, loader):
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        before_temperature_nll = nll_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f' % (before_temperature_nll))

        optimizer = optim.LBFGS([self.temperature], lr=self.lr, max_iter=1000)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        print(self.temperature)
        print('After temperature - NLL: %.3f' % (after_temperature_nll))

        return self.temperature
    

def get_imagenet_features(loader, poverty=False):
    resnet18 = torchvision.models.resnet18(pretrained=True)
    resnet18.cuda()
    resnet18.eval()

    features_all = []
    for i, batch in enumerate(loader):
        if not poverty:
            input, target = batch
        else:
            input, target, metadata = batch
            input = input[:,:3]

        # x = torch.from_numpy(np.resize(input.numpy(), (128, 3, 256, 256)))
        x = input.cuda()
        x = resnet18.conv1(x)
        x = resnet18.bn1(x)
        x = resnet18.relu(x)
        x = resnet18.maxpool(x)
        x = resnet18.layer1(x)
        x = resnet18.layer2(x)
        x = resnet18.layer3(x)
        x = resnet18.layer4(x)
        features = resnet18.avgpool(x)
        features_all.append(features.cpu().detach().numpy().squeeze())

    features_all = np.concatenate(features_all, axis=0)
    return features_all

def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Calculate the KL divergence between two univariate Gaussian distributions.
    
    Parameters:
        mu1 (float): Mean of the first Gaussian distribution.
        sigma1 (float): Standard deviation of the first Gaussian distribution.
        mu2 (float): Mean of the second Gaussian distribution.
        sigma2 (float): Standard deviation of the second Gaussian distribution.
    
    Returns:
        float: KL divergence between the two Gaussian distributions.
    """
    var1 = sigma1 ** 2
    var2 = sigma2 ** 2
    kl_div = np.log(sigma2 / sigma1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
    return kl_div
