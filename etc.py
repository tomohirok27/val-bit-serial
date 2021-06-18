import sys
import math
import argparse
import torch
import torch.nn as nn
from models.resnet import resnet18
from models.vgg import vgg16
from models.vgg11 import vgg11


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, default='vgg11')
    parser.add_argument('-d', '--dataset', type=str, default='cifar100')
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('--fixed_num', type=int, default=8)
    parser.add_argument('--fn', '--fractional_num', type=int, default=9)
    parser.add_argument('-hb', '--high_bits', type=int, default=None)
    parser.add_argument('-q',
                        '--quant',
                        type=str,
                        default='bit_serial',
                        choices=['float', 'fixed', 'bit_serial'])

    return parser.parse_args()


def get_layer_fn(model, num_bits=8):
    conv_fn_list = []
    fc_fn_list = []

    # conv layer
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            std_3 = float(m.weight.mean().abs() + 3 * m.weight.std())
            m_fn = (num_bits - 1) + round(-math.log2(std_3))
            conv_fn_list.append(m_fn)

    # fc layer
    for m in model.modules():
        if isinstance(m, nn.Linear):
            std_3 = float(m.weight.mean().abs() + 3 * m.weight.std())
            m_fn = (num_bits - 1) + round(-math.log2(std_3))
            fc_fn_list.append(m_fn)

    print(conv_fn_list)
    print(fc_fn_list)


if __name__ == '__main__':
    args = parse_args()

    device = 'cuda:' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'imagenet':
        num_classes = 1000
    else:
        print("#### unknown dataset ####")
        sys.exit()

    # floating point model
    # model = vgg11(num_classes=num_classes, quant='float')
    model = resnet18(num_classes=num_classes, quant='float')

    load_weight_path = args.weights

    if load_weight_path:
        model.load_state_dict(
            torch.load(load_weight_path, map_location='cuda:0'))

    get_layer_fn(model, args.fixed_num)
