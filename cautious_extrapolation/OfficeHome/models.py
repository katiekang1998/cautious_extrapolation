
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models



class Network(torch.nn.Module):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes):
        super(Network, self).__init__()
        self.featurizer = ResNet(input_shape) 
        
        self.classifier = torch.nn.Linear(self.featurizer.n_outputs, num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)

    def forward(self, x):
        return self.network(x)
    


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape):
        super(ResNet, self).__init__()
        self.network = torchvision.models.resnet50(pretrained=True)
        self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
    
    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()