import os
import argparse
import h5py
import numpy as np
import math
import imageio
from collections import OrderedDict

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

class Interpolate(nn.Module):
    def __init__(self, size, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        return x
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

interp1 = Interpolate([1000, 1000], 'bilinear', align_corners=True)
interp2 = Interpolate([240, 240], 'bilinear', align_corners=True)

tens1 = torch.tensor(np.array(img)[np.newaxis, np.newaxis, ...], requires_grad=False)
img1 = interp1.forward(tens1).squeeze(0)

tens2 = torch.tensor(np.moveaxis(np.array(list(f[key][:, :, :])), -1, 0)[np.newaxis, ...], requires_grad=False)
img2 = interp2.forward(tens2).squeeze(0)

a = [1,2]
b = [3,4]
a.extend(b)

a = np.random.rand(100)
b = [0, *a[0:99]]
c = np.corrcoef(a, b)

d = globals()
'''
a = OrderedDict({'b': 1, 'a': 2})
b = a.keys()
c = a.values()
123
