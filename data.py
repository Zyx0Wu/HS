import os
import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms

from utils import partition


def load_data(name, path, batch_size=32,
              # data specific arguments
              n_cells=500):
    if not os.path.isdir(path):
        os.makedirs(path)

    # HPST
    '''
    TODO: Load IR, IF data
    '''

    # MNIST
    if name is 'MNIST':
        train_data = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
        test_data = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)

        return train_data, test_data

    # PBMC
    elif name is 'PBMC':
        values = pd.read_csv(
            "%s/5k_pbmc_v3_filtered_feature_bc_matrix.csv" % path,
            usecols=range(n_cells + 1), index_col=0).dropna(axis='columns').values.astype(np.float32)
        preserves = (values[i, :] > 0 for i in range(len(values)))
        preserve = [any(tup) for tup in preserves]
        train_values = values[preserve]

        train_values = np.sqrt(train_values)
        train_values = (train_values / np.max(train_values)).astype(np.float32)
        train_labels = np.zeros(len(train_values))  # placeholder

        train_data = partition([train_values, train_labels],
                               batch_size=batch_size)
        test_data = None

        return train_data, test_data

    # else
    else:
        raise ValueError('Data name not recognized.')

