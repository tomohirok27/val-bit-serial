import sys
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import utils
import matplotlib as mpl
from conf import settings
from dataset.dataset_loader import dataset_loader
from models.resnet import resnet18
from models.vgg import vgg16


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('-m', '--model', type=str, default='vgg16')
    parser.add_argument('-d', '--dataset', type=str, default='cifar100')
    parser.add_argument('--train_batch', type=int, default=128)
    parser.add_argument('--test_batch', type=int, default=100)
    parser.add_argument('--write', type=str, default=None)
    parser.add_argument('-w', '--weights', type=str, default='/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h1_200904_1710.pth')
    parser.add_argument('--fixed_num', type=int, default=8)
    parser.add_argument('--fn', type=int, default=9)
    parser.add_argument('-hb', '--high_bits', type=int, default=1)
    parser.add_argument('-df', '--diff_fn', type=str, default=True)
    parser.add_argument('-wd', '--weight_dist', type=str, default=False)
    parser.add_argument('-q',
                        '--quant',
                        type=str,
                        default='bit_serial',
                        choices=['float', 'fixed', 'bit_serial'])
    parser.add_argument('-p', '--prominent', type=str, default='entropy')

    return parser.parse_args()


def evaluation():
    val_loss = 0
    val_acc = 0
    # ideal_bits = 7
    ideal = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0}
    model.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    model_5.eval()
    model_6.eval()
    model_7.eval()
    model_8.eval()
    with torch.no_grad():
        if args.prominent == 'entropy':
            for images, labels in tqdm(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                i_outputs = outputs
                probability = nn.functional.softmax(outputs, dim=1)
                for image, pro, i in zip(images, probability, range(len(i_outputs))):
                    ideal_bits = utils.entropy_threshold(pro)
                    ideal["%s" % ideal_bits] = ideal["%s" % ideal_bits] + 1
                    # breakpoint()
                    image_tmp = image.reshape(1, 3, 32, 32)
                    if ideal_bits == 1:
                        i_outputs[i] = model_1(image_tmp)
                    elif ideal_bits == 2:
                        i_outputs[i] = model_2(image_tmp)
                    elif ideal_bits == 3:
                        i_outputs[i] = model_3(image_tmp)
                    elif ideal_bits == 4:
                        i_outputs[i] = model_4(image_tmp)
                    elif ideal_bits == 5:
                        i_outputs[i] = model_5(image_tmp)
                    elif ideal_bits == 6:
                        i_outputs[i] = model_6(image_tmp)
                    elif ideal_bits == 7:
                        i_outputs[i] = model_7(image_tmp)
                    elif ideal_bits == 8:
                        i_outputs[i] = model_8(image_tmp)
                loss = criterion(i_outputs, labels)
                val_loss += loss.item()
                val_acc += (i_outputs.max(1)[1] == labels).sum().item()
        
        elif args.prominent == 'mahala':
            for images, labels in tqdm(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                i_outputs = outputs
                for image, output, i in zip(images, outputs, range(len(i_outputs))):
                    ideal_bits = utils.mahalanobis_distance(output)
                    ideal[ideal_bits] = ideal[ideal_bits] + 1
                    # breakpoint()
                    image_tmp = image.reshape(1, 3, 32, 32)
                    if ideal_bits == 1:
                        i_outputs[i] = model_1(image_tmp)
                    elif ideal_bits == 2:
                        i_outputs[i] = model_2(image_tmp)
                    elif ideal_bits == 3:
                        i_outputs[i] = model_3(image_tmp)
                    elif ideal_bits == 4:
                        i_outputs[i] = model_4(image_tmp)
                    elif ideal_bits == 5:
                        i_outputs[i] = model_5(image_tmp)
                    elif ideal_bits == 6:
                        i_outputs[i] = model_6(image_tmp)
                    elif ideal_bits == 7:
                        i_outputs[i] = model_7(image_tmp)
                    elif ideal_bits == 8:
                        i_outputs[i] = model_8(image_tmp)
            loss = criterion(i_outputs, labels)
            val_loss += loss.item()
            val_acc += (i_outputs.max(1)[1] == labels).sum().item()

    print('ideal_bits: %s' % ideal)
    sum = 0
    for key, value in ideal.items():
        sum += int(key) * value
    print('mean_bits: %s' % (sum / len(test_loader.dataset)))
    # breakpoint()
    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc = val_acc / len(test_loader.dataset)
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
    # load_weight_path = '/artic/t-kaneko/work/suzuki_fc_bit_serial/weights/vgg16_cifar10_ser_8_9/vgg16_cifar10_ser_8_9_200826_2130.pth'
    if load_weight_path:
        model.load_state_dict(
            torch.load(load_weight_path, map_location='cuda:0'))

    model_1 = eval(args.model)(settings=settings,
                       quant=args.quant,
                       num_classes=num_classes,
                       fixed_num=args.fixed_num,
                       fract_num=args.fn,
                       high_bits = 1,
                       diff_fn=args.diff_fn).to(device)
    model_1.load_state_dict(
            torch.load('/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h1_200904_1710.pth',
             map_location='cuda:0'))
    
    model_2 = eval(args.model)(settings=settings,
                       quant=args.quant,
                       num_classes=num_classes,
                       fixed_num=args.fixed_num,
                       fract_num=args.fn,
                       high_bits = 2,
                       diff_fn=args.diff_fn).to(device)
    model_2.load_state_dict(
            torch.load('/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h2_200909_1533.pth',
             map_location='cuda:0'))
    
    model_3 = eval(args.model)(settings=settings,
                       quant=args.quant,
                       num_classes=num_classes,
                       fixed_num=args.fixed_num,
                       fract_num=args.fn,
                       high_bits = 3,
                       diff_fn=args.diff_fn).to(device)
    model_3.load_state_dict(
            torch.load('/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h3_200909_1752.pth',
             map_location='cuda:0'))
    
    model_4 = eval(args.model)(settings=settings,
                       quant=args.quant,
                       num_classes=num_classes,
                       fixed_num=args.fixed_num,
                       fract_num=args.fn,
                       high_bits = 4,
                       diff_fn=args.diff_fn).to(device)
    model_4.load_state_dict(
            torch.load('/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h4_200909_1730.pth',
             map_location='cuda:0'))
    
    model_5 = eval(args.model)(settings=settings,
                       quant=args.quant,
                       num_classes=num_classes,
                       fixed_num=args.fixed_num,
                       fract_num=args.fn,
                       high_bits = 5,
                       diff_fn=args.diff_fn).to(device)
    model_5.load_state_dict(
            torch.load('/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h5_200909_1718.pth',
             map_location='cuda:0'))
    
    model_6 = eval(args.model)(settings=settings,
                       quant=args.quant,
                       num_classes=num_classes,
                       fixed_num=args.fixed_num,
                       fract_num=args.fn,
                       high_bits = 6,
                       diff_fn=args.diff_fn).to(device)
    model_6.load_state_dict(
            torch.load('/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h6_200909_1707.pth',
             map_location='cuda:0'))
    
    model_7 = eval(args.model)(settings=settings,
                       quant=args.quant,
                       num_classes=num_classes,
                       fixed_num=args.fixed_num,
                       fract_num=args.fn,
                       high_bits = 7,
                       diff_fn=args.diff_fn).to(device)
    model_7.load_state_dict(
            torch.load('/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h7_200909_1656.pth',
             map_location='cuda:0'))
    
    model_8 = eval(args.model)(settings=settings,
                       quant=args.quant,
                       num_classes=num_classes,
                       fixed_num=args.fixed_num,
                       fract_num=args.fn,
                       high_bits = 8,
                       diff_fn=args.diff_fn).to(device)
    model_8.load_state_dict(
            torch.load('/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h8_200909_1644.pth',
             map_location='cuda:0'))

    if args.weight_dist:
        utils.dist_loadedweght(args, model)

    _, _, test_loader = dataset_loader(dataset=args.dataset, train_batch=args.train_batch, test_batch=args.test_batch)
    
    criterion = nn.CrossEntropyLoss()

    evaluation()
