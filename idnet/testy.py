import os
import argparse
import h5py
import numpy as np
import math
import imageio

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

DATA_PATH = '../dataset/'
PLOT_PATH = '../plot/'
'''
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        DATA_PATH,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=64,
    shuffle=True,
)

for i, (imgs, _) in enumerate(dataloader):
    a = imgs[0].numpy()
    continue

a = 32 // 4

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

a = unpickle(DATA_PATH + "/cifar-10-python/cifar-10-batches-py/data_batch_1")
b = a[b'data'][0][:1024].reshape(32, -1)
c = a[b'data'][0][1024:2048].reshape(32, -1)
d = a[b'data'][0][2048:].reshape(32, -1)
'''

filename = DATA_PATH + 'hsdt/ifWT0a.hdf5'
f = h5py.File(filename, 'r')

key = list(f.keys())[0]

shape = f[key].shape

img = np.zeros(shape[:-1])
for i in range(shape[-1]):
    data = list(f[key][:, :, i])
    img += data
img /= shape[-1]
imageio.imwrite(PLOT_PATH + 'ifWT0a.png', img)

123
