'''
    Code for the custom PyTorch Dataset to fit on top of the UTK Dataset
'''

import torch
import numpy as np
from torch.utils.data import Dataset
from skimage.filters import gaussian
import pandas as pd
import os
import cv2
import skimage as sk
    
class UTKDataset(Dataset):
    def __init__(self, dataFrame, transform=None, severity=0, corruption_type="gaussian_blur"):
        # read in the transforms
        self.transform = transform
        
        # Use the dataFrame to get the pixel values
        data_holder = dataFrame.pixels.apply(lambda x: np.array(x.split(" "),dtype=float))
        arr = np.stack(data_holder)
        arr = arr / 255.0
        arr = arr.astype('float32')
        arr = arr.reshape(arr.shape[0], 48, 48, 1)
        # reshape into 48x48x1
        self.data = arr
        self.severity = severity
        self.corruption_type = corruption_type
        
        # get the age, gender, and ethnicity label arrays
        self.age_label = np.array(dataFrame.age[:])         # Note : Changed dataFrame.age to dataFrame.bins
    
    # override the length function
    def __len__(self):
        return len(self.data)
    
    # override the getitem function
    def __getitem__(self, index):
        # load the data at index and apply transform
        data = self.data[index]
        if self.severity > 0:
            if self.corruption_type == "gaussian_blur":
                data = gaussian_blur(data, self.severity)
            elif self.corruption_type == "impulse_noise":
                data = impulse_noise(data, self.severity)
        data = self.transform(data)
        
        
        # return data labels
        return data, self.age_label[index]

def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]
    x = gaussian(np.array(x), sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1)


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x), mode='s&p', amount=c)
    return np.clip(x, 0, 1)