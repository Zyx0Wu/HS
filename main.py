from __future__ import division, print_function

import os
import time
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from utils import weights_init_normal, cuda_tensors, StaticDict
from data import load_data
from model import Encoder, Decoder, Regresser
from train import train, test

# path
DATA_PATH = '../dataset'
MODEL_PATH = '../model/probtorch'
PLOTS_PATH = '../plot/probtorch'

# model parameters
INPUT_SHAPE = [817, 400, 400]
OUTPUT_SHAPE = [4, 4000, 4000]
REPRES_SHAPE = [16, *INPUT_SHAPE[1:]]
N_BLOCKS = 2
CONV_HID_CHAN = 32
SAMP_HID_CHAN = 128
MULT_HID_CHAN = 64
BATCH_NORM = False


def main(args):
    CUDA = torch.cuda.is_available() if args.cuda is None else args.cuda
    print(  # 'probtorch:', probtorch.__version__,
        'torch:', torch.__version__,
        'cuda:', CUDA)

    # initialization
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.mode is 'AESR':
        MODEL_NAME = '%s-AESR-%s' % (args.name, args.signature)
        MODES = StaticDict(OrderedDict({'train_AE': {'enc_mode': 'train', 'dec_mode': 'train'},
                                        'train_RG': {'reg_mode': 'train', 'enc_mode': 'eval'}}))
    elif args.mode is 'SR':
        MODEL_NAME = '%s-SR-%s' % (args.name, args.signature)
        MODES = StaticDict(OrderedDict({'train_SR': {'reg_mode': 'train', 'dec_mode': 'train'}}))
    else:
        raise ValueError('Model mode not recognized.')

    # load data
    kwargs = {}
    train_data, test_data = load_data(args.data, DATA_PATH, args.batch_size, **kwargs)

    # define model
    ENC = Encoder(OUTPUT_SHAPE, REPRES_SHAPE, n_blocks=N_BLOCKS,
                  conv_hid_chan=CONV_HID_CHAN, samp_hid_chan=SAMP_HID_CHAN, batch_norm=BATCH_NORM)
    DEC = Decoder(REPRES_SHAPE, OUTPUT_SHAPE, n_blocks=N_BLOCKS,
                  conv_hid_chan=CONV_HID_CHAN, samp_hid_chan=SAMP_HID_CHAN, batch_norm=BATCH_NORM)
    REG = Regresser(INPUT_SHAPE, REPRES_SHAPE, n_blocks=N_BLOCKS,
                    conv_hid_chan=CONV_HID_CHAN, mult_hid_chan=MULT_HID_CHAN, batch_norm=BATCH_NORM)

    if CUDA:
        ENC.cuda()
        DEC.cuda()
        REG.cuda()
        cuda_tensors(ENC)
        cuda_tensors(DEC)
        cuda_tensors(REG)
        '''
        ENC = nn.DataParallel(ENC)
        DEC = nn.DataParallel(DEC)
        REG = nn.DataParallel(REG)
        '''

    # initialize weights
    ENC.apply(weights_init_normal)
    DEC.apply(weights_init_normal)
    REG.apply(weights_init_normal)

    # define optimizers
    optim_dict = {'train_AE': list(ENC.parameters()) + list(DEC.parameters()),
                  'train_RG': list(REG.parameters()) + list(ENC.parameters()),
                  'train_SR': list(REG.parameters()) + list(DEC.parameters())}
    optimizers = OrderedDict()
    for mode in MODES.keys():
        optimizers[mode] = torch.optim.Adam(optim_dict[mode],
                                            lr=args.learning_rate,
                                            betas=(args.beta1, args.beta2))
    optimizers = StaticDict(optimizers)

    # train model
    if not args.restore:
        for e in range(args.num_epochs):
            train_start = time.time()
            train_loss = train(train_data, optimizers,
                               enc=ENC, dec=DEC, reg=REG,
                               batch_size=args.batch_size,
                               modes=MODES, cuda=CUDA)
            train_end = time.time()
            test_start = time.time()
            test_loss = test(test_data, enc=ENC, dec=DEC, reg=REG,
                             batch_size=args.batch_size, cuda=CUDA)
            test_end = time.time()
            if not args.mute:
                print('[Epoch %d] Train: LOSS %.4e (%ds) Test: LOSS %.4e (%ds)' % (
                        e, train_loss, train_end - train_start,
                        test_loss, test_end - test_start))
        if not args.no_sav:
            if not os.path.isdir(MODEL_PATH):
                os.mkdir(MODEL_PATH)
            torch.save(ENC.state_dict(),
                       '%s/%s-%s-ENC.rar' % (MODEL_PATH, MODEL_NAME, torch.__version__))
            torch.save(DEC.state_dict(),
                       '%s/%s-%s-DEC.rar' % (MODEL_PATH, MODEL_NAME, torch.__version__))
            torch.save(REG.state_dict(),
                       '%s/%s-%s-REG.rar' % (MODEL_PATH, MODEL_NAME, torch.__version__))

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

    parser.add_argument('--data', default='HPST', type=str,
                        help="name of the dataset to use")
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

