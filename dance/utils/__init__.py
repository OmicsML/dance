# Copyright 2022 DSE lab.  All rights reserved.
import math
import os
import random

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset


class SimpleIndexDataset(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return x


class PairedDataset(Dataset):
    """Combines two datasets into one, where input is now (x1, x2) and output is (y1,
    y2).

    A Paired dataset simply combines x and y by returning the x input and y input as a tuple, and the x output and y
    output as a tuple, and does not "cross" between the datasets

    """

    # Inherits the init from SplicedDataset since we're doing the same thing - recording
    # the two different datasets
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y

    def __len__(self):
        return len(self.dataset_x)

    def __getitem__(self, i):
        x1 = self.dataset_x[i]
        x2 = self.dataset_y[i]
        x_pair = (x1, x2)
        y_pair = (x1, x2)
        return torch.cat(x_pair), torch.cat(y_pair)


def set_seed(rndseed, cuda=True, extreme_mode=False):
    os.environ['PYTHONHASHSEED'] = str(rndseed)
    random.seed(rndseed)
    np.random.seed(rndseed)
    torch.manual_seed(rndseed)
    if cuda:
        torch.cuda.manual_seed(rndseed)
        torch.cuda.manual_seed_all(rndseed)
    if extreme_mode:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dgl.seed(rndseed)
    dgl.random.seed(rndseed)
    print('Global random seed:', rndseed)


def calculate_p(adj, l):
    adj_exp = np.exp(-1 * (adj**2) / (2 * (l**2)))
    return np.mean(np.sum(adj_exp, 1)) - 1


def search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100):
    run = 0
    p_low = calculate_p(adj, start)
    p_high = calculate_p(adj, end)
    if p_low > p + tol:
        print("l not found, try smaller start point.")
        return None
    elif p_high < p - tol:
        print("l not found, try bigger end point.")
        return None
    elif np.abs(p_low - p) <= tol:
        print("recommended l = ", str(start))
        return start
    elif np.abs(p_high - p) <= tol:
        print("recommended l = ", str(end))
        return end
    while (p_low + tol) < p < (p_high - tol):
        run += 1
        print("Run " + str(run) + ": l [" + str(start) + ", " + str(end) + "], p [" + str(p_low) + ", " + str(p_high) +
              "]")
        if run > max_run:
            print("Exact l not found, closest values are:\n" + "l=" + str(start) + ": " + "p=" + str(p_low) + "\nl=" +
                  str(end) + ": " + "p=" + str(p_high))
            return None
        mid = (start + end) / 2
        p_mid = calculate_p(adj, mid)
        if np.abs(p_mid - p) <= tol:
            print("recommended l = ", str(mid))
            return mid
        if p_mid <= p:
            start = mid
            p_low = p_mid
        else:
            end = mid
            p_high = p_mid
    return None
