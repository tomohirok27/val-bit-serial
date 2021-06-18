import os
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from distutils.util import strtobool
import utils
from conf import settings
from dataset.dataset_loader import dataset_loader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=int, default=0, help='cuda number')
    parser.add_argument('-m', '--model', type=str, default='resnet18')
    parser.add_argument('-d', '--dataset', type=str, default='cifar100')
    parser.add_argument('--train_batch', type=int, default=128)
    parser.add_argument('--test_batch', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('-e', '--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1, help='leraning rate')
    parser.add_argument('-ss', '--step_size', type=int, default=60)
    parser.add_argument('-g', '--gamma', type=float, default=0.2)
    parser.add_argument('--writer', type=str, default=False)
    parser.add_argument('--csv', type=str, default=False)
    parser.add_argument('-cfn', '--csv_file_name', type=str, default='out.csv')
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('-sw', '--save_weight', type=str, default='False')
    parser.add_argument('--fixed_num', type=int, default=8)
    parser.add_argument('--fn', '--fractional_num', type=int, default=9)
    parser.add_argument('-hb', '--high_bits', type=int, default=None)
    parser.add_argument('-df', '--diff_fn', type=str, default=False)
    parser.add_argument('-ds', '--data_select', type=str, default='pollux01')
    parser.add_argument('-q',
                        '--quant',
                        type=str,
                        default='bit_serial',
                        choices=['float', 'fixed', 'bit_serial'])

    return parser.parse_args()


def train(epoch, val_ratio):
    train_loss = 0
    train_acc = 0

    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        breakpoint()
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        acc = (outputs.max(1)[1] == labels).sum()
        train_acc += acc.item()
        loss.backward()
        optimizer.step()

        if args.writer:
            step = epoch * len(train_loader) + i
            writer.add_scalar('Train/Loss', loss.item(), step)
            writer.add_scalar('Train/Accuracy', acc.item(), step)

    avg_train_loss = train_loss / (len(train_loader.dataset)*(1 - val_ratio))
    # avg_train_acc = train_acc / len(train_loader.dataset)
    scheduler.step()

    return avg_train_loss


def evaluation(epoch, train_loss, val_ratio):
    val_loss = 0
    val_acc = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            acc = (outputs.max(1)[1] == labels).sum()
            val_acc += acc.item()

    avg_val_loss = val_loss / (len(train_loader.dataset)*val_ratio)
    avg_val_acc = val_acc / (len(train_loader.dataset)*val_ratio)
    print(
        'Epoch [{}/{}], LR: {LR:.5f}, Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
        .format(epoch + 1,
                args.num_epochs,
                LR=lr,
                loss=train_loss,
                val_loss=avg_val_loss,
                val_acc=avg_val_acc))

    if args.writer:
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        writer.add_scalar('Val/Accuracy', avg_val_acc, epoch)

    return avg_val_acc

if __name__ == '__main__':
    args = parse_args()

    device = 'cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu'
    model = utils.get_model(args, device, settings)
    val_ratio = 0.2

    load_weight_path = args.weights
    if load_weight_path:
        model.load_state_dict(
            torch.load(load_weight_path, map_location='cuda:0'))

    if args.high_bits:
        print("##### freeze weight except BN #####")
        for p in model.parameters():
            p.requires_grad = False

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = True


    save_weight_path, file_name, model_name  = utils.get_save_weight_path(args)

    
    train_loader, val_loader, test_loader = dataset_loader(args.dataset, args.train_batch,
                                               args.test_batch,
                                               args.num_worker,
                                               args.data_select,
                                               val_ratio)

    if args.writer:
        date = datetime.datetime.now().strftime("%y%m%d_%H%M")
        tensorboard_path = 'runs/%s/%s' % (model_name, date)
        writer = SummaryWriter(tensorboard_path)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(),
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
        momentum=0.9,
        # weight_decay=1e-4)
        weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size,
                                          args.gamma)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 100],
    #                                            gamma=args.gamma)

    avg_train_loss = 0.
    avg_train_acc = 0.
    acc_max = 0.

    for epoch in range(args.num_epochs):
        lr = optimizer.param_groups[0]['lr']
        # training
        train_loss = train(epoch, val_ratio)
        # # evaluation
        val_acc = evaluation(epoch, train_loss, val_ratio)

        if val_acc > acc_max:
            acc_max = val_acc
            # save weight
            if strtobool(args.save_weight):
                os.makedirs(save_weight_path, exist_ok=True)
                torch.save(model.state_dict(), save_weight_path + file_name)

        # if args.writer * (not bool(args.high_bits)):
        if args.writer:
            utils.add_param(writer, model, epoch)

    if args.csv:
        csv_name = args.csv_file_name
        csv_dir = 'csv'
        csvfile = '%s/%s' % (csv_dir, csv_name)
        os.makedirs(csv_dir, exist_ok=True)
        utils.csv_write(csvfile, file_name[:-4], acc_max,args.train_batch,args.test_batch,args.num_worker,args.num_epochs,args.lr,
        args.step_size,args.gamma,args.fixed_num,args.data_select)
        print("#### write results to csv file ####")

    print("max accuracy =", acc_max)
    if args.writer:
        writer.close()
