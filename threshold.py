import sys
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import utils

W1 = '201217/201217_1527_resnet18_cifar100_ser_8_h1_201214_1941.npy'
W2 = '201217/201217_1528_resnet18_cifar100_ser_8_h2_201214_2014.npy'
W3 = '201217/201217_1529_resnet18_cifar100_ser_8_h3_201214_2046.npy'
W4 = '201217/201217_1529_resnet18_cifar100_ser_8_h4_201214_2057.npy'
W5 = '201217/201217_1530_resnet18_cifar100_ser_8_h5_201214_2108.npy'
W6 = '201217/201217_1530_resnet18_cifar100_ser_8_h6_201214_2119.npy'
W7 = '201217/201217_1531_resnet18_cifar100_ser_8_h7_201214_2131.npy'
W8 = '201217/201217_1532_resnet18_cifar100_ser_8_h8_201214_2142.npy'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--data_select', type=str, default='pollux01')
    parser.add_argument('-dn', '--data_npy', type=int, default=0)
    parser.add_argument('-sn', '--sort_npy', type=int, default=0)
    parser.add_argument('-pc', '--plt_cum', type=str, default=None)
    parser.add_argument('-pe', '--plt_entropy', type=str, default=None)
    parser.add_argument('-tt', '--train_th', type=float, default=0.99)

    return parser.parse_args()

def npy_loader():
    if args.data_npy:
        if args.data_select == 'pollux01':
            if args.data_npy == 1:
                data_file = W1
            elif args.data_npy == 2:
                data_file = W2
            elif args.data_npy == 3:
                data_file = W3
            elif args.data_npy == 4:
                data_file = W4
            elif args.data_npy == 5:
                data_file = W5
            elif args.data_npy == 6:
                data_file = W6
            elif args.data_npy == 7:
                data_file = W7
            elif args.data_npy == 8:
                data_file = W8
        elif args.data_select == 'zeus01':
            if args.data_npy == 1:
                data_file = '201215/201215_2232_resnet18_cifar100_ser_8_h1_201215_0105.npy'
            elif args.data_npy == 2:
                data_file = '201215/201215_2233_resnet18_cifar100_ser_8_h2_201215_0134.npy'
            elif args.data_npy == 3:
                data_file = '201215/201215_2233_resnet18_cifar100_ser_8_h3_201215_0202.npy'
            elif args.data_npy == 4:
                data_file = '201215/201215_2234_resnet18_cifar100_ser_8_h4_201215_0211.npy'
            elif args.data_npy == 5:
                data_file = '201215/201215_2234_resnet18_cifar100_ser_8_h5_201215_0221.npy'
            elif args.data_npy == 6:
                data_file = '201215/201215_2235_resnet18_cifar100_ser_8_h6_201215_0230.npy'
            elif args.data_npy == 7:
                data_file = '201215/201215_2236_resnet18_cifar100_ser_8_h7_201215_0239.npy'
            elif args.data_npy == 8:
                data_file = '201215/201215_2236_resnet18_cifar100_ser_8_h8_201215_0249.npy'
        elif args.data_select == 'selene01':
            if args.data_npy == 1:
                data_file = '201215/201215_2257_resnet18_cifar100_ser_8_h1_201215_0353.npy'
            elif args.data_npy == 2:
                data_file = '201215/201215_2257_resnet18_cifar100_ser_8_h2_201215_0422.npy'
            elif args.data_npy == 3:
                data_file = '201215/201215_2258_resnet18_cifar100_ser_8_h3_201215_0450.npy'
            elif args.data_npy == 4:
                data_file = '201215/201215_2259_resnet18_cifar100_ser_8_h4_201215_0500.npy'
            elif args.data_npy == 5:
                data_file = '201215/201215_2259_resnet18_cifar100_ser_8_h5_201215_0510.npy'
            elif args.data_npy == 6:
                data_file = '201215/201215_2300_resnet18_cifar100_ser_8_h6_201215_0520.npy'
            elif args.data_npy == 7:
                data_file = '201215/201215_2300_resnet18_cifar100_ser_8_h7_201215_0529.npy'
            elif args.data_npy == 8:
                data_file = '201215/201215_2301_resnet18_cifar100_ser_8_h8_201215_0539.npy'

        all_value = np.load('/artic/t-kaneko/work/val_bit_serial/numpy/data/%s/%s' % (args.data_select, data_file))
        sort = np.zeros_like(all_value)
        sort = all_value[:, np.argsort(all_value[0])]
        filename = utils.numpy_save(sort, args.data_select, data_file, 'sort')
    if args.sort_npy:
        if args.data_select == 'pollux01':
            if args.sort_npy == 1:
                filename = '201215/201215_2153_201215_2047_resnet18_cifar100_ser_8_h1_201214_1941.npy'
            elif args.sort_npy == 2:
                filename = '201215/201215_2153_201215_2048_resnet18_cifar100_ser_8_h2_201214_2014.npy'
            elif args.sort_npy == 3:
                filename = '201215/201215_2153_201215_2049_resnet18_cifar100_ser_8_h3_201214_2046.npy'
            elif args.sort_npy == 4:
                filename = '201215/201215_2153_201215_2050_resnet18_cifar100_ser_8_h4_201214_2057.npy'
            elif args.sort_npy == 5:
                filename = '201215/201215_2153_201215_2051_resnet18_cifar100_ser_8_h5_201214_2108.npy'
            elif args.sort_npy == 6:
                filename = '201215/201215_2153_201215_2052_resnet18_cifar100_ser_8_h6_201214_2119.npy'
            elif args.sort_npy == 7:
                filename = '201215/201215_2153_201215_2053_resnet18_cifar100_ser_8_h7_201214_2131.npy'
            elif args.sort_npy == 8:
                filename = '201215/201215_2153_201215_2054_resnet18_cifar100_ser_8_h8_201214_2142.npy'
        elif args.data_select == 'zeus01':
            if args.sort_npy == 1:
                filename = '201215/201215_2244_201215_2232_resnet18_cifar100_ser_8_h1_201215_0105.npy'
            elif args.sort_npy == 2:
                filename = '201215/201215_2244_201215_2233_resnet18_cifar100_ser_8_h2_201215_0134.npy'
            elif args.sort_npy == 3:
                filename = '201215/201215_2244_201215_2233_resnet18_cifar100_ser_8_h3_201215_0202.npy'
            elif args.sort_npy == 4:
                filename = '201215/201215_2244_201215_2234_resnet18_cifar100_ser_8_h4_201215_0211.npy'
            elif args.sort_npy == 5:
                filename = '201215/201215_2244_201215_2234_resnet18_cifar100_ser_8_h5_201215_0221.npy'
            elif args.sort_npy == 6:
                filename = '201215/201215_2245_201215_2235_resnet18_cifar100_ser_8_h6_201215_0230.npy'
            elif args.sort_npy == 7:
                filename = '201215/201215_2245_201215_2236_resnet18_cifar100_ser_8_h7_201215_0239.npy'
            elif args.sort_npy == 8:
                filename = '201215/201215_2245_201215_2236_resnet18_cifar100_ser_8_h8_201215_0249.npy'
        elif args.data_select == 'selene01':
            if args.sort_npy == 1:
                filename = '201215/201215_2304_201215_2257_resnet18_cifar100_ser_8_h1_201215_0353.npy'
            elif args.sort_npy == 2:
                filename = '201215/201215_2304_201215_2257_resnet18_cifar100_ser_8_h2_201215_0422.npy'
            elif args.sort_npy == 3:
                filename = '201215/201215_2304_201215_2258_resnet18_cifar100_ser_8_h3_201215_0450.npy'
            elif args.sort_npy == 4:
                filename = '201215/201215_2304_201215_2259_resnet18_cifar100_ser_8_h4_201215_0500.npy'
            elif args.sort_npy == 5:
                filename = '201215/201215_2304_201215_2259_resnet18_cifar100_ser_8_h5_201215_0510.npy'
            elif args.sort_npy == 6:
                filename = '201215/201215_2304_201215_2300_resnet18_cifar100_ser_8_h6_201215_0520.npy'
            elif args.sort_npy == 7:
                filename = '201215/201215_2304_201215_2300_resnet18_cifar100_ser_8_h7_201215_0529.npy'
            elif args.sort_npy == 8:
                filename = '201215/201215_2304_201215_2301_resnet18_cifar100_ser_8_h8_201215_0539.npy'

        sort = np.load('/artic/t-kaneko/work/val_bit_serial/numpy/sort/%s/%s' % (args.data_select, filename))

    return filename, sort

if __name__ == '__main__':
    args = parse_args()

    filename, sort = npy_loader()
    cum =  np.zeros_like(sort)
    # cum[0] = sort[0]
    cum[1] = np.cumsum(sort[1])
    cum[2] = np.cumsum(sort[2])
    correct_rate = np.empty(0)
    for i, (x, y) in enumerate(zip(cum[1], cum[2])):
        r = x / (x + y)
        correct_rate = np.append(correct_rate, r)
        if r > args.train_th:
            key = i
            ratio = r

    if args.data_npy:
        high_bits = args.data_npy
    if args.sort_npy:
        high_bits = args.sort_npy
    if args.plt_cum:
        utils.plt_cum(cum[1], cum[2], correct_rate, high_bits, 'frequency', args.data_select, filename, key, ratio, sort[0][key], args.train_th)
    
    if args.plt_entropy:
        utils.plt_number_entropy(sort, high_bits, 'entropy', args.data_select, filename)
    # breakpoint()
    utils.correct_rate_entropy(correct_rate, sort[0], high_bits, filename, 'entropy_and_frequenct', args.data_select)
    if args.data_npy == 1 or args.sort_npy == 1:
        print('1bit: %s' % sort[0][key])
    else:
        print('%sbit: %s' % (args.data_npy or args.sort_npy, sort[0][(np.where(sort[2]==1)[0][0])]))