

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

from  torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import numpy as np
from cautious_extrapolation.data_paths import DATA_PATHS

class ImageNet200(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        classes, class_to_idx = self.find_classes(DATA_PATHS["nfs"]["ImageNet-R"])
        samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS, None)

        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples