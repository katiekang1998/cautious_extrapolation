  # camera-ready

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms


import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import math
import scipy.stats

import pickle

import cv2
import os

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, data_path):
        with open(os.path.join(data_path, "labels_train.pkl"), "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open(os.path.join(data_path, "images_train.pkl"), "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)
        self.labels-= 1005.2143
        self.labels/=706.77203

        self.num_examples = self.labels.shape[0]

        print ("DatasetTrain - number of images: %d" % self.num_examples)
        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = img.astype(np.float32)

        return (img, label)

    def __len__(self):
        return self.num_examples

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, data_path):
        with open(os.path.join(data_path, "labels_val.pkl"), "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open(os.path.join(data_path, "images_val.pkl"), "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        self.labels-= 1005.2143
        self.labels/=706.77203

        self.num_examples = self.labels.shape[0]

        print ("DatasetVal - number of images: %d" % self.num_examples)
        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = img.astype(np.float32)

        return (img, label)

    def __len__(self):
        return self.num_examples

class DatasetTest(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # with open(os.path.join(data_path, "labels_test.pkl"), "rb") as file: # (needed for python3)
        #     self.labels = pickle.load(file)
        # with open(os.path.join(data_path, "images_test.pkl"), "rb") as file: # (needed for python3)
        #     self.imgs = pickle.load(file)
        with open(os.path.join(data_path, "labels_val.pkl"), "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open(os.path.join(data_path, "images_val.pkl"), "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)
        self.labels-= 1005.2143
        self.labels/=706.77203

        self.num_examples = self.labels.shape[0]

        print ("DatasetTest - number of images: %d" % self.num_examples)
        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64, 3))

        # img = speckle_noise(img, severity=5)

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = img.astype(np.float32)

        return (img, label)

    def __len__(self):
        return self.num_examples


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255