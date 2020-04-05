import numpy as np

import torch.nn as nn

from utils import Interpolate, MultiscaleConv2d

class Encoder(nn.Module):
    def __init__(self, input_shape, output_shape, n_blocks=16,
                 conv_hid_chan=64, samp_hid_chan=256, batch_norm=False):
        super(Encoder, self).__init__()
        self.n_blocks = n_blocks

        input_size = input_shape[1:]
        input_chan = input_shape[0]
        output_size = output_shape[1:]
        output_chan = output_shape[0]

        def samp_unit(in_size, out_size, in_channels, out_channels):
            times = np.int(np.log(out_size/in_size)/np.log(2))

            unit = [nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)]
            for i in range(times):
                unit.extend([nn.LeakyReLU(0.2, inplace=True),
                             nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)])
            unit.extend([Interpolate(out_size, 'bilinear', align_corners=True),
                         nn.LeakyReLU(0.2, inplace=True)])
            return nn.Sequential(*unit)

        def conv_block(in_filters, out_filters, reps=1, batch_norm=False):
            block = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1)]
            for i in range(reps):
                if batch_norm:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                block.extend([nn.LeakyReLU(0.2, inplace=True),
                              nn.Conv2d(out_filters, out_filters, 3, stride=1, padding=1)])
                if batch_norm:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.samp_unit = samp_unit(input_size, output_size, input_chan, samp_hid_chan)

        self.conn_unit = nn.Sequential(nn.Conv2d(samp_hid_chan, conv_hid_chan, 3, stride=1, padding=1),
                                       nn.LeakyReLU(0.2, inplace=True))

        if n_blocks > 0:
            self.conv_blocks = nn.ModuleList()
            for k in range(n_blocks):
                self.conv_blocks.append(nn.Sequential(*conv_block(conv_hid_chan, conv_hid_chan,
                                                                  reps=1, batch_norm=batch_norm)))

        self.conc_unit = [nn.Conv2d(conv_hid_chan, conv_hid_chan, 3, stride=1, padding=1)]
        if batch_norm:
            self.conc_unit.append(nn.BatchNorm2d(conv_hid_chan, 0.8))
        self.conc_unit = nn.Sequential(*self.conc_unit)

        self.last_unit = nn.Sequential(nn.Conv2d(conv_hid_chan, output_chan, 1, stride=1))

    def forward(self, y):
        z = self.samp_unit.forward(y)
        z = self.conn_unit.forward(z)
        temp = z
        if self.n_blocks > 0:
            for block in self.conv_blocks:
                z = block.forward(z) + z
        z = self.conc_unit.forward(z) + temp
        z = self.last_unit.forward(z)
        return z


class Decoder(nn.Module):
    def __init__(self, input_shape, output_shape, n_blocks=16,
                 conv_hid_chan=64, samp_hid_chan=256, batch_norm=False):
        super(Decoder, self).__init__()
        self.n_blocks = n_blocks

        input_size = input_shape[1:]
        input_chan = input_shape[0]
        output_size = output_shape[1:]
        output_chan = output_shape[0]

        def conv_block(in_filters, out_filters, reps=1, batch_norm=False):
            block = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1)]
            for i in range(reps):
                if batch_norm:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                block.extend([nn.LeakyReLU(0.2, inplace=True),
                              nn.Conv2d(out_filters, out_filters, 3, stride=1, padding=1)])
                if batch_norm:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        def samp_unit(in_size, out_size, in_channels, out_channels):
            times = np.int(np.log(out_size/in_size)/np.log(2))

            unit = [nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)]
            for i in range(times):
                unit.extend([nn.PixelShuffle(2),
                             nn.LeakyReLU(0.2, inplace=True),
                             nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)])
            unit.extend([Interpolate(out_size, 'bilinear', align_corners=True),
                         nn.LeakyReLU(0.2, inplace=True)])
            return nn.Sequential(*unit)

        self.init_unit = nn.Sequential(nn.Conv2d(input_chan, conv_hid_chan, 3, stride=1, padding=1),
                                       nn.LeakyReLU(0.2, inplace=True))

        if n_blocks > 0:
            self.conv_blocks = nn.ModuleList()
            for k in range(n_blocks):
                self.conv_blocks.append(nn.Sequential(*conv_block(conv_hid_chan, conv_hid_chan,
                                                                  reps=1, batch_norm=batch_norm)))

        self.conc_unit = [nn.Conv2d(conv_hid_chan, conv_hid_chan, 3, stride=1, padding=1)]
        if batch_norm:
            self.conc_unit.append(nn.BatchNorm2d(conv_hid_chan, 0.8))
        self.conc_unit = nn.Sequential(*self.conc_unit)

        self.samp_unit = samp_unit(input_size, output_size, conv_hid_chan, samp_hid_chan)

        self.last_unit = nn.Sequential(nn.Conv2d(samp_hid_chan, output_chan, 1, stride=1))

    def forward(self, z):
        y = self.init_unit.forward(z)
        temp = y
        if self.n_blocks > 0:
            for block in self.conv_blocks:
                y = block.forward(y) + y
        y = self.conc_unit.forward(y) + temp
        y = self.samp_unit.forward(y)
        y = self.last_unit.forward(y)
        return y


class Regresser(nn.Module):
    def __init__(self, input_shape, output_shape, n_blocks=16,
                 conv_hid_chan=64, mult_hid_chan=128, batch_norm=False):
        super(Regresser, self).__init__()
        self.n_blocks = n_blocks

        input_size = input_shape[1:]
        input_chan = input_shape[0]
        output_size = output_shape[1:]
        output_chan = output_shape[0]

        def conv_block(in_filters, out_filters, reps=1, batch_norm=False):
            block = [nn.Conv2d(in_filters, out_filters, 1, stride=1)]
            for i in range(reps):
                if batch_norm:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                block.extend([nn.LeakyReLU(0.2, inplace=True),
                              nn.Conv2d(out_filters, out_filters, 1, stride=1)])
                if batch_norm:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        #self.aggr_unit = nn.Sequential(nn.Conv3d(1, 1, (5, 1, 1), stride=(2, 1, 1), padding=(2, 0, 0)))

        self.mult_unit = nn.Sequential(MultiscaleConv2d(input_chan, mult_hid_chan),
                                       nn.LeakyReLU(0.2, inplace=True))

        self.conn_unit = nn.Sequential(nn.Conv2d(3*mult_hid_chan, conv_hid_chan,
                                                 tuple(2*(input_size-output_size)+1), stride=1),
                                       nn.LeakyReLU(0.2, inplace=True))

        if n_blocks > 0:
            self.conv_blocks = nn.ModuleList()
            for k in range(n_blocks):
                self.conv_blocks.append(nn.Sequential(*conv_block(conv_hid_chan, conv_hid_chan,
                                                                  reps=1, batch_norm=batch_norm)))

        self.conc_unit = [nn.Conv2d(conv_hid_chan, conv_hid_chan, 3, stride=1, padding=1)]
        if batch_norm:
            self.conc_unit.append(nn.BatchNorm2d(conv_hid_chan, 0.8))
        self.conc_unit = nn.Sequential(*self.conc_unit)

        self.last_unit = nn.Sequential(nn.Conv2d(conv_hid_chan, output_chan, 1, stride=1))

    def forward(self, x):
        z = self.mult_unit.forward(x)
        z = self.conn_unit.forward(z)
        temp = z
        if self.n_blocks > 0:
            for block in self.conv_blocks:
                z = block.forward(z) + z
        z = self.conc_unit.forward(z) + temp
        z = self.last_unit.forward(z)

        return z

