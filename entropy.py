import sys
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import utils
from conf import settings
from dataset.dataset_loader import dataset_loader
from models.resnet import resnet18
from models.vgg import vgg16
from models.modules.serial_bit import SBReLU





def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('-m', '--model', type=str, default='vgg16')
    parser.add_argument('-d', '--dataset', type=str, default='cifar100')
    parser.add_argument('--train_batch', type=int, default=128)
    parser.add_argument('--test_batch', type=int, default=100)
    parser.add_argument('-w', '--weights', type=str,default='/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h1_200904_1710.pth')
    parser.add_argument('--fixed_num', type=int, default=8)
    parser.add_argument('--fn', type=int, default=9)
    parser.add_argument('-hb', '--high_bits', type=int, default=1)
    parser.add_argument('-df', '--diff_fn', type=str, default=True)
    parser.add_argument('-wd', '--weight_dist', type=str, default=False)
    parser.add_argument('-p', '--prominent', type=str, default='entropy')
    parser.add_argument('-se', '--save_entropy', type=str, default=None)
    parser.add_argument('-sp', '--save_pro', type=str, default=None)
    parser.add_argument('-sl', '--save_label', type=str, default=None)
    parser.add_argument('-ds', '--data_select', type=str, default='pollux01')
    parser.add_argument('-q',
                        '--quant',
                        type=str,
                        default='bit_serial',
                        choices=['float', 'fixed', 'bit_serial'])

    return parser.parse_args()


def evaluation():
    val_loss = 0
    val_acc = 0
    all_entropy = np.empty(0)
    all_pro = []
    all_label = []
    correct = np.empty(0)
    incorrect = np.empty(0)
    model.eval()
    with torch.no_grad():
        if args.prominent == 'entropy':
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                probability = nn.functional.softmax(outputs, dim=1)
                # breakpoint()
                for (pro, output, label) in zip(probability, outputs, labels):
                    # entropy = 0
                    # breakpoint()
                    all_pro.append(pro)
                    all_label.append(label)
                    # for p in pro:
                    #     entropy += - p * torch.log2(p)
                    # all_entropy = np.append(all_entropy, entropy.to('cpu').detach().numpy().copy())
                    # correct = np.append(correct, (1 if output.max(0)[1] == label else 0))
                    # incorrect = np.append(incorrect, (0 if output.max(0)[1] == label else 1))

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()

        elif args.prominent == 'mahalanobis':
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                for (output, label) in zip(outputs, labels):
                    mahalanobis = ((output.max() - output.mean()).abs()) / output.std()
                    # df.loc[count] = (mahalanobis, (1 if output.max(0)[1] == label else 0), (0 if output.max(0)[1] == label else 1))
                    # count += 1
                       
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
    
    # all_value = np.stack([all_entropy, correct, incorrect])
    if args.save_entropy:
        utils.numpy_save(all_value, args.data_select, args.weights, 'data')
    if args.save_pro:
        utils.numpy_save(all_pro, args.data_select, args.weights, 'pro')
    if args.save_label:
        utils.numpy_save(all_label, args.data_select, args.weights, 'label')

    avg_val_loss = val_loss / (len(train_loader.dataset)*0.2)
    avg_val_acc = val_acc / (len(train_loader.dataset)*0.2)
    print()
    print('val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'.format(
        val_loss=avg_val_loss, val_acc=avg_val_acc))


if __name__ == '__main__':
    args = parse_args()
    device = 'cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'imagenet':
        num_classes = 1000
    else:
        print("#### unknown dataset ####")
        sys.exit()
    
    model = eval(args.model)(settings=settings,
                       quant=args.quant,
                       num_classes=num_classes,
                       fixed_num=args.fixed_num,
                       fract_num=args.fn,
                       high_bits=args.high_bits,
                       diff_fn = args.diff_fn).to(device)

    load_weight_path = args.weights
    # load_weight_path = '/artic/t-kaneko/work/suzuki_fc_bit_serial/weights/vgg16_cifar10_ser_8_h1_9/vgg16_cifar10_ser_8_h1_9_200826_2130.pth'
    if load_weight_path:
        model.load_state_dict(
            torch.load(load_weight_path, map_location='cuda:0'))


    if args.weight_dist:
        utils.dist_loadedweght(args, model)
    torch.manual_seed(0)
    train_loader, val_loader, test_loader = dataset_loader(dataset=args.dataset, train_batch=args.train_batch, data_name=args.data_select,test_batch=args.test_batch)
    
    criterion = nn.CrossEntropyLoss()

    evaluation()
    
