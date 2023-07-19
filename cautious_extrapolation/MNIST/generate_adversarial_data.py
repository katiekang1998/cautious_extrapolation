from __future__ import print_function
import argparse
from train import Net
from torch.utils.data import Dataset
import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from cautious_extrapolation.data_paths import DATA_PATHS


dir_path = os.path.dirname(os.path.realpath(__file__))
model = Net().to("cuda")
model.load_state_dict(torch.load(os.path.join(dir_path, "data", "model1.pt")))
model.eval()
model.cuda()

class P_MNIST(Dataset):
    def __init__(self, perturbed_imgs, labels, transform=None):
        self.transform = transform
        self.imgs = perturbed_imgs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])



def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    return perturbed_image

def test(model, test_loader, epsilon):
    perturbed_data_all = np.zeros((10000, 1, 28, 28))
    labels_all = np.zeros((10000))
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.cuda().float(), target.cuda().long()
        data.requires_grad = True
        output = model(data)
        output = torch.e**output
        init_pred = output.argmax(dim=1) # get the index of the max log-probability
        init_accuracy = (init_pred == target).float().mean()
        init_entropy = -(output*torch.clip(torch.log(output), min=-1000000)).sum(dim=1).mean()
        # loss = -F.nll_loss(output, (target+1)%10)
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)


        perturbed_data_all[batch_idx*1024:(batch_idx+1)*1024] = perturbed_data.cpu().detach().numpy()
        labels_all[batch_idx*1024:(batch_idx+1)*1024] = target.cpu().detach().numpy()
        output = model(perturbed_data)
        output = torch.e**output
        final_pred = output.argmax(dim=1) # get the index of the max log-probability
        final_accuracy = (final_pred == target).float().mean()
        final_entropy = -(output*torch.clip(torch.log(output), min=-1000000)).sum(dim=1).mean()
        print(init_accuracy.item(), init_entropy.item(), final_accuracy.item(), final_entropy.item())

    return perturbed_data_all, labels_all


test_dataset = datasets.MNIST(DATA_PATHS["desktop"]["MNIST"], train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024)



for _ in range(20):
    print(_)
    if _ == 0:
        purturbed_test_loader = test_loader
    else:
        purturbed_test_loader = torch.utils.data.DataLoader(P_MNIST(perturbed_data_all, labels_all), batch_size=1024)
    perturbed_data_all, labels_all = test(model, purturbed_test_loader, 0.1)
    perturbed_ds = {}
    perturbed_ds["image"] = perturbed_data_all
    perturbed_ds["label"] = labels_all
    file = open(os.path.join(dir_path, "data", "perturbed_ds_"+str(_)+".pkl"),'wb')
    pickle.dump(perturbed_ds, file)
    file.close()