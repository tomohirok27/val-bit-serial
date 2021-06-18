import sys
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import dist_loadedweght
from conf import settings
from dataset.dataset_loader import dataset_loader
from models.resnet import resnet18
from models.vgg11 import vgg11
from models.modules.serial_bit import SBReLU





def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('-m', '--model', type=str, default='vgg16')
    parser.add_argument('-d', '--dataset', type=str, default='cifar100')
    parser.add_argument('--write', type=str, default=None)
    parser.add_argument('-w', '--weights', type=str, default='/artic/t-kaneko/work/bit_serial/weights/vgg11_cifar100_ser_8_h4/vgg11_cifar100_ser_8_h4_201118/vgg11_cifar100_ser_8_h4_201118_1741.pth')
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

    return parser.parse_args()


def evaluation():
    val_loss = 0
    val_acc = 0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()

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

    if args.model == 'resnet18':
        model = resnet18(settings=settings,
                     quant=args.quant,
                     num_classes=num_classes,
                     fixed_num=args.fixed_num,
                     fract_num=args.fn,
                     high_bits=args.high_bits,
                     diff_fn=args.diff_fn).to(device)
    elif args.model == 'vgg11':
        model = vgg11(settings=settings,
                     quant=args.quant,
                     num_classes=num_classes,
                     fixed_num=args.fixed_num,
                     fract_num=args.fn,
                     high_bits=args.high_bits,
                     diff_fn=args.diff_fn).to(device)

    load_weight_path = args.weights
    # load_weight_path = '/artic/t-kaneko/work/suzuki_fc_bit_serial/weights/vgg11_cifar10_ser_8_9/vgg11_cifar10_ser_8_9_200826_2130.pth'
    if load_weight_path:
        model.load_state_dict(
            torch.load(load_weight_path, map_location='cuda:0'))
            # torch.load(load_weight_path, map_location=torch.device('cpu')))
    
    if args.weight_dist:
        dist_loadedweght(args, model)

 
    _, test_loader = dataset_loader(args.dataset)
    

    criterion = nn.CrossEntropyLoss()

    evaluation()
    
