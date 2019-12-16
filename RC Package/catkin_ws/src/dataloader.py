from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import utils
import torchvision.transforms as transforms
import cv2
from PIL import Image

def read_video(filename, root_dir):
    #create frames folder in root_dir
    path = os.path.join(root_dir, "frames")
    dataset = []
    vid = cv2.VideoCapture(filename)
    sucess = True
    success,image = vid.read()
    count = 0
    while success:
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # cv2.imshow("image", grayscale)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(path, "frame%d.jpg" % count), image)
        count += 1
        success,image = vid.read()

    return path, count

class AVData(Dataset):
    def __init__(self, dataset, transform, root_dir):
        self.dataset = dataset
        self.root_dir = root_dir
        self.frame_dir, self.len = read_video(dataset, root_dir)
        self.steering = np.random.randint(1,101,len(self))
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        s = Image.open(os.path.join(self.frame_dir, "frame%d.jpg" % index))
        return self.transform(s), self.steering[index]
