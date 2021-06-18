"""" data loader """ ""

import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import random


def dataset_loader(dataset="None",
                   train_batch=128,
                   test_batch=100,
                   num_worker=2,
                   data_name='selene01',
                   val_ratio=0.2):
    # random.seed(1)
    return eval(dataset + '_loader')(train_batch, test_batch, num_worker, data_name, val_ratio)
    # if dataset == "cifer10":
    #     return cirfer10_loader()
    # if dataset == "cifer100":
    #     return cirfer100_loader()

    # else:
    #     print("no such dataset!!!")
    #     print("no such dataset!!!")
    #     sys.exit(1)


def cifar10_loader(train_batch=128, test_batch=100, num_workers=2):
    """ cifer10 loader """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
        # transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
        # transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
    ])

    dir_path = os.path.dirname(__file__) + '/data'
    # print(dir_path)

    trainval_dataset = torchvision.datasets.CIFAR10(root=dir_path,
                                                 train=True,
                                                 transform=train_transform,
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=dir_path,
                                                train=False,
                                                transform=test_transform,
                                                download=True)
    n_samples = len(trainval_dataset) 
    train_size = int(len(trainval_dataset) * 0.8)# train_size is 48000
    val_size = n_samples - train_size 

    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_batch,
                                               shuffle=True,
                                               num_workers=num_workers)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=test_batch,
                                               shuffle=False,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch,
                                              shuffle=False,
                                              num_workers=num_workers)
    # breakpoint()

    return train_loader, val_loader, test_loader


def cifar100_loader(train_batch=128, test_batch=128, num_workers=2, data_name='selene01', val_ratio=0.2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.438, 0.418, 0.377), (0.300, 0.287, 0.294))
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])


    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.438, 0.418, 0.377), (0.300, 0.287, 0.294))
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])

    dir_path = os.path.dirname(__file__) + '/data'

    train_dataset = torchvision.datasets.CIFAR100(root=dir_path,
                                                  train=True,
                                                  transform=train_transform,
                                                  download=True)
    test_dataset = torchvision.datasets.CIFAR100(root=dir_path,
                                                 train=False,
                                                 transform=test_transform,
                                                 download=True)
    
    breakpoint()
    train_set_index = torch.randperm(len(train_dataset))
    # breakpoint()
    save_name = '/artic/t-kaneko/work/val_bit_serial/data/%s' % data_name
    if os.path.exists('%s_index.pth' % save_name):
        print('!!!!!! Load  %s_index.pth !!!!!!' % os.path.basename(save_name))
        train_set_index = torch.load('%s_index.pth' % save_name)
    else:
        print('!!!!!! Save %s_index.pth !!!!!!' % os.path.basename(save_name))
        torch.save(train_set_index, '%s_index.pth' % save_name)
    # n_samples = len(trainval_dataset) 
    # train_size = int(len(trainval_dataset) * 0.8)
    # val_size = n_samples - train_size 
    
    # train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
    num_sample_valid = int(len(train_dataset)*val_ratio)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=train_batch,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(train_set_index[:-num_sample_valid]),
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                             batch_size=test_batch,
                                             sampler=torch.utils.data.sampler.SubsetRandomSampler(train_set_index[-num_sample_valid:]),
                                             num_workers=num_workers)

    # breakpoint()
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=train_batch,
    #                                            shuffle=True,
    #                                            num_workers=num_workers)
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                            batch_size=test_batch,
    #                                            shuffle=True,
    #                                            num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch,
                                              shuffle=False,
                                              num_workers=num_workers)
    # breakpoint()
    return train_loader, val_loader, test_loader

def imagenet_loader(train_batch=128, test_batch=100, num_workers=2):
    """ ImageNet loader """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
        # transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
        # transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
    ])

    # dir_path = os.path.dirname(__file__) + '/data'
    # print(dir_path)
    dir_path = '../../Shared/Datasets/ILSVRC'

    trainval_dataset = torchvision.datasets.ImageFolder(root=dir_path,
                                                transform=train_transform
                                                )
    test_dataset = torchvision.datasets.ImageFolder(root=dir_path,
                                                transform=test_transform
                                                )

    n_samples = len(trainval_dataset) 
    train_size = int(len(trainval_dataset) * 0.8)
    val_size = n_samples - train_size 

    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_batch,
                                               shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=test_batch,
                                               shuffle=True,
                                               num_workers=num_workers)                                               
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch,
                                              shuffle=False,
                                              num_workers=num_workers)

    return train_loader, val_loader, test_loader