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

class IDNetwork(nn.Module):
    def __init__(self, num_input,
                 num_hidden,
                 num_output,
                 temperature=0.33,
                 eps=1e-9):
        super(self.__class__, self).__init__()
        self.probs = nn.Parameter(0.9 * torch.ones(num_input, requires_grad=True))
        self.temperature = temperature
        if type(num_hidden) == int:
            self.hidden = nn.Sequential(
                nn.Linear(num_input, num_hidden),
                nn.BatchNorm1d(num_hidden),
                nn.LeakyReLU())
            self.output = nn.Sequential(
                nn.Linear(num_hidden, num_output),
                nn.Softmax(dim=-1))
        else:
            self.hidden = nn.ModuleList()
            self.hidden.append(
                nn.Sequential(
                    nn.Linear(num_input, num_hidden[0]),
                    nn.BatchNorm1d(num_hidden[0]),
                    nn.LeakyReLU()))
            for k in range(len(num_hidden) - 1):
                self.hidden.append(
                    nn.Sequential(
                        nn.Linear(num_hidden[k], num_hidden[k+1]),
                        nn.BatchNorm1d(num_hidden[k+1]),
                        nn.LeakyReLU()))
            self.output = nn.Sequential(
                nn.Linear(num_hidden[-1], num_output),
                nn.Softmax(dim=-1))
        self.eps = eps

    @expand_inputs
    def forward(self, data, labels, num_samples=None):
        if num_samples is None:
            indep_shape, dep_dims = [data.shape[0:1], slice(1, data.ndimension())]
        else:
            indep_shape, dep_dims = [data.shape[0:2], slice(2, data.ndimension())]
        q = probtorch.Trace()
        probs_sig = self.probs
        for d in reversed(range(len(indep_shape))):
            probs_sig = probs_sig.unsqueeze(0).expand(indep_shape[d], *probs_sig.shape)
        signs = q.variable(RelaxedBernoulli,
                           probs=probs_sig,
                           temperature=self.temperature,
                           name='signs')
        inputs = signs * data
        hiddens = inputs.reshape((-1, *inputs.shape[dep_dims]))
        for m in self.hidden:
            hiddens = m(hiddens)
        outputs = self.output(hiddens)
        probs_lab = outputs.reshape((*indep_shape, *outputs.shape[1:]))
        q.loss(lambda x_hat, x: -(torch.log(x_hat + self.eps) * x +
                                  torch.log(1 - x_hat + self.eps) * (1-x)).sum(-1),
               probs_lab, labels, name='classes')
        return q

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
    WEIGHTS_PATH = '../model/probtorch'
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
    idnet = IDNetwork(NUM_PIXELS, NUM_HIDDEN, NUM_DIGITS, temperature=args.temperature)

    if CUDA:
        idnet.cuda()
        cuda_tensors(idnet)
        idnet = nn.DataParallel(idnet)

    optimizer = torch.optim.Adam(list(idnet.parameters()),
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
            if not os.path.isdir(WEIGHTS_PATH):
                os.mkdir(WEIGHTS_PATH)
            torch.save(idnet.state_dict(),
                       '%s/%s-%s-%s-idnet.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__))

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

