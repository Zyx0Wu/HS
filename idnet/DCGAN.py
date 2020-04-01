from __future__ import division, print_function

import os
import sys
import argparse
import time
from datetime import datetime

import math
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.distributions import RelaxedBernoulli, OneHotCategorical, Normal, MultivariateNormal
from torchvision import datasets, transforms
import probtorch
from probtorch.util import expand_inputs
from probtorch.objectives.montecarlo import ml

###

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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


class Encoder(nn.Module):
    def __init__(self, input_shape, output_shape, n_blocks=16,
                 conv_hid_chan=64, samp_hid_chan=256, batch_norm=False):
        super(Encoder, self).__init__()
        self.n_blocks = n_blocks

        input_size = input_shape[2, 3]
        input_chan = input_shape[1]
        output_size = output_shape[2, 3]
        output_chan = output_shape[1]

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

    def forward(self, x):
        z = self.samp_unit.forward(x)
        z = self.conn_unit.forward(z)
        temp = z
        if self.n_blocks > 0:
            for block in self.conv_blocks:
                z = block.forward(z) + z
        z = self.conc_unit.forward(z) + temp
        z = self.last_unit.forward(z)
        return z

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

class Decoder(nn.Module):
    def __init__(self, input_shape, output_shape, n_blocks=16,
                 conv_hid_chan=64, samp_hid_chan=256, batch_norm=False):
        super(Decoder, self).__init__()
        self.n_blocks = n_blocks

        input_size = input_shape[2, 3]
        input_chan = input_shape[1]
        output_size = output_shape[2, 3]
        output_chan = output_shape[1]

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

    def forward(self, x):
        z = self.init_unit.forward(x)
        temp = z
        if self.n_blocks > 0:
            for block in self.conv_blocks:
                z = block.forward(z) + z
        z = self.conc_unit.forward(z) + temp
        z = self.samp_unit.forward(z)
        z = self.last_unit.forward(z)
        return z


class Regresser(nn.Module):
    def __init__(self, input_shape, output_shape, n_blocks=16,
                 conv_hid_chan=64, mult_hid_chan=128, batch_norm=False):
        super(Regresser, self).__init__()
        self.n_blocks = n_blocks

        input_size = input_shape[2, 3]
        input_chan = input_shape[1]
        output_size = output_shape[2, 3]
        output_chan = output_shape[1]

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

###

def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())

def train(data, num_classes, network, optimizer,
          num_samples=32, batch_size=16, alpha=1.,
          eps=1e-9, cuda=False):
    epoch_loss = 0.0
    network.train()
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == batch_size:
            # count
            N += batch_size
            # reshape
            images = images.reshape(images.shape[0], -1)
            labels_onehot = torch.zeros(batch_size, num_classes)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, eps, 1-eps)
            # cuda
            if cuda:
                images = images.cuda()
                labels_onehot = labels_onehot.cuda()
            # run
            optimizer.zero_grad()
            q = network(images, labels_onehot, num_samples=num_samples)
            # loss
            log_weights = q.log_joint(0, 1, q.conditioned())
            probs_sig = q['signs'].dist.probs
            loss = - ml(q, 0, 1, log_weights, size_average=True, reduce=True) \
                   + alpha * F.l1_loss(probs_sig, torch.zeros_like(probs_sig), reduction='mean')
            # step
            loss.backward()
            optimizer.step()
            # cuda
            if cuda:
                loss = loss.cpu()
            # add
            epoch_loss += loss.item()
    return epoch_loss / N

def test(data, network,
         num_samples=32, batch_size=16, alpha=1.,
         cuda=False):
    network.eval()
    epoch_loss = 0.0
    epoch_correct = 0
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == batch_size:
            N += batch_size
            images = images.reshape(images.shape[0], -1)
            if cuda:
                images = images.cuda()
            q = network(images, num_samples=num_samples)
            log_weights = q.log_joint(0, 1, q.conditioned())
            batch_loss = - ml(q, 0, 1, log_weights, size_average=True, reduce=True) \
                         + alpha * F.l1_loss(q['signs'].probs, reduction='mean')
            if cuda:
                batch_loss = batch_loss.cpu()
            epoch_loss += batch_loss.item()
            _, y_pred = q['classes'].value.max(-1)
            if cuda:
                y_pred = y_pred.cpu()
            epoch_correct += (labels == y_pred).sum().item() / (num_samples or 1.0)
    return epoch_loss / N, epoch_correct / N

###

def main(args):
    # initialization
    CUDA = torch.cuda.is_available() if args.cuda is None else args.cuda
    print('probtorch:', probtorch.__version__,
          'torch:', torch.__version__,
          'cuda:', CUDA)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # model parameters
    NUM_PIXELS = 784
    NUM_HIDDEN = 256
    NUM_DIGITS = 10

    # path
    MODEL_NAME = 'mnist-idnetwork-%s' % args.signature
    DATA_PATH = '../dataset'
    MODEL_PATH = '../model/probtorch'
    PLOTS_PATH = '../plot/probtorch'

    # load data
    if not os.path.isdir(DATA_PATH):
        os.makedirs(DATA_PATH)

    train_data = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)
    test_data = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)

    # define model
    generator = Generator()
    discriminator = Discriminator()

    if CUDA:
        generator.cuda()
        discriminator.cuda()
        cuda_tensors(generator)
        cuda_tensors(discriminator)
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # define optimizer
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=args.learning_rate,
                                   betas=(args.beta1, args.beta2))
    optimizer_D = torch.optim.Adam(generator.parameters(),
                                   lr=args.learning_rate,
                                   betas=(args.beta1, args.beta2))

    # learn model
    if not args.restore:
        for e in range(args.num_epochs):
            train_start = time.time()
            train_loss, mask = train(train_data, NUM_DIGITS, idnet, optimizer,
                                     num_samples=args.num_samples, batch_size=args.batch_size, alpha=args.alpha,
                                     eps=args.epsilon, cuda=CUDA)

            train_end = time.time()
            test_start = time.time()
            test_loss, test_accuracy = test(test_data, idnet,
                                            num_samples=args.num_samples, batch_size=args.batch_size, alpha=args.alpha,
                                            cuda=CUDA)
            test_end = time.time()
            if not args.mute:
                print('[Epoch %d] Train: LOSS %.4e (%ds) Test: LOSS %.4e, Accuracy %0.3f (%ds)' % (
                        e, train_loss, train_end - train_start,
                        test_loss, test_accuracy, test_end - test_start))
        if not args.no_sav:
            if not os.path.isdir(MODEL_PATH):
                os.mkdir(MODEL_PATH)
            torch.save(idnet.state_dict(),
                       '%s/%s-%s-%s-idnet.rar' % (MODEL_PATH, MODEL_NAME, probtorch.__version__, torch.__version__))

    # save identifiers
    if CUDA:
        probs = idnet.probs.cpu().detach().numpy().reshape((28, 28))
    else:
        probs = idnet.probs.detach().numpy().reshape((28, 28))
    np.savetxt('%s/%s-idnet-probs.txt' % (MODEL_PATH, MODEL_NAME), probs, delimiter=',')

    # analyze result
    if not args.no_vis:
        if not os.path.isdir(PLOTS_PATH):
            os.mkdir(PLOTS_PATH)
        '''TODO'''
###

EXAMPLE_RUN = "example run: python ssvae_rnaseq_probtorch.py --cuda False --seed 0 " \
              "--temperature 0.33 --alpha 1.0 --beta1 0.90 --beta2 0.999 " \
              "-e 10 -s 8 -b 16 -lr 1e-4 -eps 1e-9 --signature example"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SS-VAE\n{}".format(EXAMPLE_RUN))

    parser.add_argument('--cuda', default=None, type=bool,
                        help="whether to use GPU(s) for training")
    parser.add_argument('--seed', default=0, type=int,
                        help="seed for controlling randomness")
    parser.add_argument('--temperature', default=0.33, type=float,
                        help="temperature of relaxed one-hot distribution")
    parser.add_argument('--alpha', default=0.0, type=float,
                        help="relative importance of classification loss")
    parser.add_argument('--beta1', default=0.90, type=float,
                        help="beta1 of Adam optimizer")
    parser.add_argument('--beta2', default=0.999, type=float,
                        help="beta2 of Adam optimizer")
    parser.add_argument('-e', '--num-epochs', default=5, type=int,
                        help="number of epochs to run")
    parser.add_argument('-s', '--num-samples', default=8, type=int,
                        help="number of samples to draw")
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help="batch size")
    parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float,
                        help="learning rate of the network")
    parser.add_argument('-eps', '--epsilon', default=1e-9, type=float,
                        help="epsilon")
    parser.add_argument('--restore', action='store_true',
                        help="restore")
    parser.add_argument('--mute', action='store_true',
                        help="mute outputs")
    parser.add_argument('--no-sav', action='store_true',
                        help="do not save model")
    parser.add_argument('--no-vis', action='store_true',
                        help="do not plot result")
    parser.add_argument('--signature', default="default", type=str,
                        help="signature for the outputs")

    args = parser.parse_args()

    main(args)

