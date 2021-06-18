#!/bin/bash

#selene01
W='/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8/resnet18_cifar100_ser_8_201214/resnet18_cifar100_ser_8_201214_1851.pth'

python3 main.py -w $W --lr 0.0001 -m resnet18 --csv True --writer True -sw True -df True --high_bits 1 -cfn out_2012150352_ser_8h_selene01.csv -ds selene01 -e 60
python3 main.py -w $W --lr 0.0001 -m resnet18 --csv True --writer True -sw True -df True --high_bits 2 -cfn out_2012150352_ser_8h_selene01.csv -ds selene01 -e 60
python3 main.py -w $W --lr 0.0001 -m resnet18 --csv True --writer True -sw True -df True --high_bits 3 -cfn out_2012150352_ser_8h_selene01.csv -ds selene01 -e 20
python3 main.py -w $W --lr 0.0001 -m resnet18 --csv True --writer True -sw True -df True --high_bits 4 -cfn out_2012150352_ser_8h_selene01.csv -ds selene01 -e 20
python3 main.py -w $W --lr 0.0001 -m resnet18 --csv True --writer True -sw True -df True --high_bits 5 -cfn out_2012150352_ser_8h_selene01.csv -ds selene01 -e 20
python3 main.py -w $W --lr 0.0001 -m resnet18 --csv True --writer True -sw True -df True --high_bits 6 -cfn out_2012150352_ser_8h_selene01.csv -ds selene01 -e 20
python3 main.py -w $W --lr 0.0001 -m resnet18 --csv True --writer True -sw True -df True --high_bits 7 -cfn out_2012150352_ser_8h_selene01.csv -ds selene01 -e 20
python3 main.py -w $W --lr 0.0001 -m resnet18 --csv True --writer True -sw True -df True --high_bits 8 -cfn out_2012150352_ser_8h_selene01.csv -ds selene01 -e 20

# for i in `seq 8`
# do
#     python3 main.py -w $W --lr 0.0001 -m resnet18 --csv True --writer True -sw True -df True --high_bits $i -cfn out_2012141938_ser_8h_pollux01.csv -ds zeus01
# done
# python3 main.py  --lr 0.001 -m resnet18 --weights weights/resnet18_cifar100_ser_8/resnet18_cifar100_ser_8_201206/resnet18_cifar100_ser_8_201206_1623.pth --csv True --writer True -sw True -df True --high_bits 1
# python3 main.py  --lr 0.001 -m resnet18 --weights weights/resnet18_cifar100_ser_8/resnet18_cifar100_ser_8_201206/resnet18_cifar100_ser_8_201206_1623.pth --csv True --writer True -sw True -df True --high_bits 2
# python3 main.py  --lr 0.001 -m resnet18 --weights weights/resnet18_cifar100_ser_8/resnet18_cifar100_ser_8_201206/resnet18_cifar100_ser_8_201206_1623.pth --csv True --writer True -sw True -df True --high_bits 3
# python3 main.py  --lr 0.001 -m resnet18 --weights weights/resnet18_cifar100_ser_8/resnet18_cifar100_ser_8_201206/resnet18_cifar100_ser_8_201206_1623.pth --csv True --writer True -sw True -df True --high_bits 4
# python3 main.py  --lr 0.001 -m resnet18 --weights weights/resnet18_cifar100_ser_8/resnet18_cifar100_ser_8_201206/resnet18_cifar100_ser_8_201206_1623.pth --csv True --writer True -sw True -df True --high_bits 5
# python3 main.py  --lr 0.001 -m resnet18 --weights weights/resnet18_cifar100_ser_8/resnet18_cifar100_ser_8_201206/resnet18_cifar100_ser_8_201206_1623.pth --csv True --writer True -sw True -df True --high_bits 6
# python3 main.py  --lr 0.001 -m resnet18 --weights weights/resnet18_cifar100_ser_8/resnet18_cifar100_ser_8_201206/resnet18_cifar100_ser_8_201206_1623.pth --csv True --writer True -sw True -df True --high_bits 7
# python3 main.py  --lr 0.001 -m resnet18 --weights weights/resnet18_cifar100_ser_8/resnet18_cifar100_ser_8_201206/resnet18_cifar100_ser_8_201206_1623.pth --csv True --writer True -sw True -df True --high_bits 8





# python3 main.py  --lr 0.001 -m vgg11 --weights weights/vgg11_cifar100_ser_8/vgg11_cifar100_ser_8_201117/vgg11_cifar100_ser_8_201117_2340.pth --csv True --writer True -sw True -df True --high_bits 7
# python3 main.py  --lr 0.001 -m vgg11 --weights weights/vgg11_cifar100_ser_8/vgg11_cifar100_ser_8_201117/vgg11_cifar100_ser_8_201117_2340.pth --csv True --writer True -sw True -df True --high_bits 6
# python3 main.py  --lr 0.001 -m vgg11 --weights weights/vgg11_cifar100_ser_8/vgg11_cifar100_ser_8_201117/vgg11_cifar100_ser_8_201117_2340.pth --csv True --writer True -sw True -df True --high_bits 5
# python3 main.py  --lr 0.001 -m vgg11 --weights weights/vgg11_cifar100_ser_8/vgg11_cifar100_ser_8_201117/vgg11_cifar100_ser_8_201117_2340.pth --csv True --writer True -sw True -df True --high_bits 4
# python3 main.py  --lr 0.001 -m vgg11 --weights weights/vgg11_cifar100_ser_8/vgg11_cifar100_ser_8_201117/vgg11_cifar100_ser_8_201117_2340.pth --csv True --writer True -sw True -df True --high_bits 3
# python3 main.py  --lr 0.001 -m vgg11 --weights weights/vgg11_cifar100_ser_8/vgg11_cifar100_ser_8_201117/vgg11_cifar100_ser_8_201117_2340.pth --csv True --writer True -sw True -df True --high_bits 2
# python3 main.py  --lr 0.001 -m vgg11 --weights weights/vgg11_cifar100_ser_8/vgg11_cifar100_ser_8_201117/vgg11_cifar100_ser_8_201117_2340.pth --csv True --writer True -sw True -df True --high_bits 1

