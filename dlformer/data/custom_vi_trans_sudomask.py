import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from dlformer.data.base_vi import ImagePaths_trans_sudomask_orishape as ImagePaths


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, h, w, training_images_list_file, training_mask_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        with open(training_mask_file, "r") as f:
            paths_mask = f.read().splitlines()
        self.data = ImagePaths(paths=paths, paths_mask=paths_mask, h=h, w=w, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, h, w, test_images_list_file, test_mask_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        with open(test_mask_file, "r") as f:
            paths_mask = f.read().splitlines()
        self.data = ImagePaths(paths=paths, paths_mask=paths_mask, h=h, w=w, random_crop=False)


class CustomTest_specific(CustomBase):
    def __init__(self, h=240, w=432, test_images_list_file='./data/mydata/camel.txt', test_mask_file='./data/mydata/camel_mask.txt'):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        with open(test_mask_file, "r") as f:
            paths_mask = f.read().splitlines()
        self.data = ImagePaths(paths=paths, paths_mask=paths_mask, h=h, w=w, random_crop=False)
