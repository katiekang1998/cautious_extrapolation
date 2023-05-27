import torch
import os
import torch.nn as nn

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



def get_rl_loss(num_actions, misspecification_cost=4):
    def rl_loss(output, target_var):
        one_hot = nn.functional.one_hot(target_var.to(torch.int64), num_actions)
        reward = (misspecification_cost+1)*one_hot - misspecification_cost
        reward[:, -1] = 0
        loss = torch.mean((output-reward)**2)
        return loss
    return rl_loss