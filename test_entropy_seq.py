import sys
import argparse
import csv
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import utils
from conf import settings
from dataset.dataset_loader import dataset_loader
from models.resnet import resnet18

W8 = '/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h8_200909_1644.pth'
W7 = '/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h7_200909_1656.pth'
W6 = '/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h6_200909_1707.pth'
W5 = '/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h5_200909_1718.pth'
W4 = '/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h4_200909_1730.pth'
W3 = '/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h3_200909_1752.pth'
W2 = '/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h2_200909_1533.pth'
W1 = '/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h1_200904_1710.pth'

orig1_1 = '/artic/t-kaneko/work/val_bit_serial/numpy/data/pollux01/201217/201217_1527_resnet18_cifar100_ser_8_h1_201214_1941.npy'
orig1_2 = '/artic/t-kaneko/work/val_bit_serial/numpy/data/pollux01/201217/201217_1528_resnet18_cifar100_ser_8_h2_201214_2014.npy'
orig1_3 = '/artic/t-kaneko/work/val_bit_serial/numpy/data/pollux01/201217/201217_1529_resnet18_cifar100_ser_8_h3_201214_2046.npy'
orig1_4 = '/artic/t-kaneko/work/val_bit_serial/numpy/data/pollux01/201217/201217_1529_resnet18_cifar100_ser_8_h4_201214_2057.npy'
orig1_5 = '/artic/t-kaneko/work/val_bit_serial/numpy/data/pollux01/201217/201217_1530_resnet18_cifar100_ser_8_h5_201214_2108.npy'
orig1_6 = '/artic/t-kaneko/work/val_bit_serial/numpy/data/pollux01/201217/201217_1530_resnet18_cifar100_ser_8_h6_201214_2119.npy'
orig1_7 = '/artic/t-kaneko/work/val_bit_serial/numpy/data/pollux01/201217/201217_1531_resnet18_cifar100_ser_8_h7_201214_2131.npy'
orig1_8 = '/artic/t-kaneko/work/val_bit_serial/numpy/data/pollux01/201217/201217_1532_resnet18_cifar100_ser_8_h8_201214_2142.npy'

#selene01
# W1 = '/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h1/resnet18_cifar100_ser_8_h1_201215/resnet18_cifar100_ser_8_h1_201215_0353.pth'
# W2 = '/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h2/resnet18_cifar100_ser_8_h2_201215/resnet18_cifar100_ser_8_h2_201215_0422.pth'
# W3 = '/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h3/resnet18_cifar100_ser_8_h3_201215/resnet18_cifar100_ser_8_h3_201215_0450.pth'
# W4 = '/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h4/resnet18_cifar100_ser_8_h4_201215/resnet18_cifar100_ser_8_h4_201215_0500.pth'
# W5 = '/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h5/resnet18_cifar100_ser_8_h5_201215/resnet18_cifar100_ser_8_h5_201215_0510.pth'
# W6 = '/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h6/resnet18_cifar100_ser_8_h6_201215/resnet18_cifar100_ser_8_h6_201215_0520.pth'
# W7 = '/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h7/resnet18_cifar100_ser_8_h7_201215/resnet18_cifar100_ser_8_h7_201215_0529.pth'
# W8 = '/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h8/resnet18_cifar100_ser_8_h8_201215/resnet18_cifar100_ser_8_h8_201215_0539.pth'

exep = 5.5
exep_bit = 4
# th_list = [0.52868247, 1.38352299, 2.63476491]   # th_var = 0.25
# th_list = [0.28, 0.9, 1.67]               # th_std = 0.4
# th_list = [0.52868247, 1.38352299, 2.63476491]   # std = 0.5
# th_list = [0.62173206, 1.62679851, 3.10947967]   # std = 0.54
# th_list = [0.64786613, 1.66506076, 3.22332239, exep]   # var = 0.30
# th_list = [0.90191227, 2.15162373, 4.90269852]   # std = 0.62
# th_list = [0.20466846, 0.54944319, 1.18805826, 1.57448828] #th_std = 1/3
# th_list = [0.20466846, 0.54944319, 1.18805826, 4] #th_std = 1/3
# th_list = [2.1, 3, 6]
# th_list = [1.2, 1.8, 6]
# th_difficult = 5.308468545506302
# th_difficult = 5.82012706207698
# th_difficult  = 5.756169747505646
# th_difficult  = 6.011999005790985
# th_difficult  = 5.244511230934967
# th_difficult = 5.116596601792298
# th_difficult  = 6.267828264076324
# th_ee = 4.6
# th1 = 0
# th2 = 100
# th3 = 4.5
# th_p = 1.0
# th_e = 2
NUM = exep_bit - 1
acc_list = []
avg_bit_list = []
bit_1 = []
bit_2 = []
bit_3 = []
bit_4 = []


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('-m', '--model', type=str, default='resnet18')
    parser.add_argument('-d', '--dataset', type=str, default='cifar100')
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('--write', type=str, default=None)
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('--fixed_num', type=int, default=8)
    parser.add_argument('--fn', type=int, default=9)
    parser.add_argument('-hb', '--high_bits', type=int, default=None)
    parser.add_argument('-df', '--diff_fn', type=str, default=True)
    parser.add_argument('-q',
                        '--quant',
                        type=str,
                        default='bit_serial',
                        choices=['float', 'fixed', 'bit_serial'])

    return parser.parse_args()


def entropy(a):
    # breakpoint()
    return (torch.log2(a) * (-a)).sum()


def softmax(a):
    return torch.exp(a) / torch.exp(a).sum()


def evaluation(th_list):
    val_loss = 0
    val_acc = 0
    for i in range(8):
        model_list[i].eval()
    # model.eval()
    cnt_list = np.zeros([len(th_list)+1], dtype=int)    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = torch.zeros([args.batch_size, num_classes]).to(device)
            # tmp = torch.zeros([len(th_list), args.batch_size]).to(device)
            # bit = torch.zeros([args.batch_size], dtype=int)
            tmp_data = np.zeros([len(th_list),args.batch_size, 2])
            # entropy
            for i in range(len(th_list)):
                output_Nbit = model_list[i](images)
                # outputs = torch.zeros_like(output_Nbit)
                for j in range(len(output_Nbit)):
                    if outputs[j].abs().max() == 0.:
                        tmp_soft = softmax(output_Nbit[j])
                        # breakpoint()
                        tmp_data[i][j][0] = tmp_soft.argmax()
                        E = entropy(tmp_soft)
                        tmp_data[i][j][1] = E
                        # tmp[i][j] = E
                        # image = images[j].reshape(1, 3, 32, 32)
                        if E < th_list[i]:
                            # num = i
                            outputs[j] = output_Nbit[j]
                            cnt_list[i] += 1
                        # if i == 3:
                            # breakpoint()
                            # if E > th_difficult:
                            #     outputs[j] = model_list[0](images[j].reshape(1, 3, 32, 32))
                            #     cnt_list[0] += 1
                            # bit[j] = i
                        # break
                    # num = NUM
                    # num = NUM
                    # for j in range(len(th_list)):
                    #     if E < th_list[j]:
                    #         num = j
                    #         break
            output_Nbit = model_list[NUM](images)
            cnt_list[NUM] += len(torch.where(outputs.sum(dim=1) == 0)[0])
            # for j in range(len(torch.where(outputs.sum(dim=1) == 0)[0])):
            #     key = torch.where(outputs.sum(dim=1) == 0)[0][j]
            #     tmp_soft4bit = softmax(output_Nbit[key])
            #     E = entropy(tmp_soft4bit)
            #     if len(set([tmp_data[0][key][0], tmp_data[1][key][0], tmp_data[2][key][0], np.argmax(tmp_soft4bit.to('cpu').detach().numpy().copy())])) >= 3:
            #         if (E>th_ee) and (tmp_data[0][key][1]>th_ee) and (tmp_data[1][key][1]>th_ee) and (tmp_data[2][key][1]>th_ee):
            #             output_Nbit[key] = model_list[0](images[j].reshape(1, 3, 32, 32))
            #             cnt_list[0] += 1
            #         else:
            #             cnt_list[NUM] += 1
            #     else:
            #         cnt_list[NUM] += 1
                # if tmp_data[0][key][0] == tmp_soft4bit.argmax():
                #     if th_e < tmp_data[0][key][1]:
                #         if (tmp_data[0][key][1] * th_p) < E:
                #             output_Nbit[key] = model_list[0](images[j].reshape(1, 3, 32, 32))
                #             cnt_list[0] += 1
                #         else:
                #             cnt_list[NUM] += 1
                #     else:
                #         cnt_list[NUM] += 1
                # else:
                #     cnt_list[NUM] += 1

                # if E > th_difficult:
                #     output_Nbit[key] = model_list[0](images[j].reshape(1, 3, 32, 32))
                #     cnt_list[0] += 1
                # else:
                #     cnt_list[NUM] += 1

                 
            # for k in torch.where(outputs.sum(dim=1) == 0)[0]:
            #     bit[k] = 3
            outputs = torch.where(outputs == 0., output_Nbit, outputs)
            # for j in range(len(output_Nbit)):
            #     if (tmp[0][j] - tmp[1][j]).abs() < 0.2:
            #         image = images[j].reshape(1, 3, 32, 32)
            #         outputs[j] = model_list[0](image)
            #         cnt_list[bit[j]] -= 1
            #         cnt_list[0] += 1
            # cnt_list = np.where(cnt_list == 0, len(th_list), cnt_list)

                # outputs[i] = model_list[num](image)
                # cnt_list[num] += 1
                
            # outputs = model_list[0](images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()

    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc = val_acc / len(test_loader.dataset)
    print()
    print('val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'.format(
        val_loss=avg_val_loss, val_acc=avg_val_acc))
    # print(cnt1, cnt2, cnt3, cnt4)
    # print(
    #     (cnt1 + 2 * cnt2 + 3 * cnt3 + 4 * cnt4) / (cnt1 + cnt2 + cnt3 + cnt4))
    print(cnt_list)
    sum_bit_len = 0.
    for i in range(len(cnt_list)):
        sum_bit_len += cnt_list[i] * (i + 1)
    mean_bit = sum_bit_len / len(test_loader.dataset)
    print(mean_bit)

    return avg_val_acc, mean_bit, cnt_list



    
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

    model_8 = resnet18(settings=settings,
                       num_classes=100,
                       fixed_num=8,
                       fract_num=9,
                       high_bits=8,
                       diff_fn=args.diff_fn).to(device)
    model_8.load_state_dict(torch.load(W8, map_location='cuda:0'))

    model_7 = resnet18(settings=settings,
                       num_classes=100,
                       fixed_num=8,
                       fract_num=9,
                       high_bits=7,
                       diff_fn=args.diff_fn).to(device)
    model_7.load_state_dict(torch.load(W7, map_location='cuda:0'))

    model_6 = resnet18(settings=settings,
                       num_classes=100,
                       fixed_num=8,
                       fract_num=9,
                       high_bits=6,
                       diff_fn=args.diff_fn).to(device)
    model_6.load_state_dict(torch.load(W6, map_location='cuda:0'))

    model_5 = resnet18(settings=settings,
                       num_classes=100,
                       fixed_num=8,
                       fract_num=9,
                       high_bits=5,
                       diff_fn=args.diff_fn).to(device)
    model_5.load_state_dict(torch.load(W5, map_location='cuda:0'))

    model_4 = resnet18(settings=settings,
                       num_classes=100,
                       fixed_num=8,
                       fract_num=9,
                       high_bits=4,
                       diff_fn=args.diff_fn).to(device)
    model_4.load_state_dict(torch.load(W4, map_location='cuda:0'))

    model_3 = resnet18(settings=settings,
                       num_classes=100,
                       fixed_num=8,
                       fract_num=9,
                       high_bits=3,
                       diff_fn=args.diff_fn).to(device)
    model_3.load_state_dict(torch.load(W3, map_location='cuda:0'))

    model_2 = resnet18(settings=settings,
                       num_classes=100,
                       fixed_num=8,
                       fract_num=9,
                       high_bits=2,
                       diff_fn=args.diff_fn).to(device)
    model_2.load_state_dict(torch.load(W2, map_location='cuda:0'))

    model_1 = resnet18(settings=settings,
                       num_classes=100,
                       fixed_num=8,
                       fract_num=9,
                       high_bits=1,
                       diff_fn=args.diff_fn).to(device)
    model_1.load_state_dict(torch.load(W1, map_location='cuda:0'))

    model_list = [
        model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8
    ]

    _, _,  test_loader = dataset_loader(args.dataset, test_batch=args.batch_size)
    criterion = nn.CrossEntropyLoss()

    orig_list = []
    for i in range(5):
        exec('orig = np.load(orig1_%s)' % (i + 1))
        orig_list.append(orig)

    en_sort_list = []
    for j in range(len(orig_list)):
        tmp = orig_list[j][:, orig_list[j][0].argsort()]
        tmp[1] = tmp[1].cumsum()
        for i in range(len(tmp[1])):
            tmp[1][i] = tmp[1][i] / (i + 1)
        en_sort_list.append(tmp)
    
    for i in range(100, 500, 100):
        alpha = 1 / i
        th_list = utils.threshold(alpha, en_sort_list)
        avg_val_acc, mean_bit, cnt_list = evaluation(th_list)
        print(th_list)
        print(i)
        # with open('csv/diff300_210127_thee49_argmax3.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([avg_val_acc, mean_bit, cnt_list])
        #     writer.writerow([th_list,i])
        acc_list .append(avg_val_acc)
        avg_bit_list.append(mean_bit)
        bit_1.append(cnt_list[0])
        bit_2.append(cnt_list[1])
        bit_3.append(cnt_list[2])
        bit_4.append(cnt_list[3])
    # val_acc = evaluation(th_list)

    # np.save('numpy/list_data/pollux01/acc_list210118', np.array(acc_list))
    # np.save('numpy/list_data/pollux01/avg_bit_list210118', np.array(avg_bit_list))
    # np.save('numpy/list_data/pollux01/bit1_1210118', np.array(bit_1))
    # np.save('numpy/list_data/pollux01/bit2_1210118', np.array(bit_2))
    # np.save('numpy/list_data/pollux01/bit3_1210118', np.array(bit_3))
    # np.save('numpy/list_data/pollux01/bit4_1210118', np.array(bit_4))



