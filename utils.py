import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticDict:
    def __init__(self, srcdict):
        self._srcdict = srcdict

    def __getitem__(self, idx):
        return self._srcdict[idx]

    def keys(self):
        return self._srcdict.keys()

###


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Interpolate(nn.Module):
    def __init__(self, size, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        return x


class MultiscaleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MultiscaleConv2d, self).__init__()
        self.Conv2dScale1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0,
                                      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.Conv2dScale3 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1,
                                      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.Conv2dScale5 = nn.Conv2d(in_channels, out_channels, 5, stride=1, padding=2,
                                      dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        x1 = self.Conv2dScale1.forward(x)
        x3 = self.Conv2dScale1.forward(x)
        x5 = self.Conv2dScale1.forward(x)
        x = torch.cat((x1, x3, x5), 1)
        return x


###


def partition(data, batch_size, to_tensor=True):
    data_size = len(data[0])
    index = np.random.permutation(data_size)
    n_batch = math.ceil(data_size / batch_size)

    part_data = [None] * n_batch
    for i in range(n_batch):
        dat = [None] * len(data)
        for j in range(len(data)):
            dat[j] = data[j][index[i*batch_size:
                                   min((i+1)*batch_size, data_size)]]
            if to_tensor:
                dat[j] = torch.tensor(dat[j], requires_grad=False)
        part_data[i] = tuple(dat)
    return part_data

def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())

