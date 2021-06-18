#!/bin/bash

W1='/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h1/resnet18_cifar100_ser_8_h1_201214/resnet18_cifar100_ser_8_h1_201214_1941.pth'
W2='/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h2/resnet18_cifar100_ser_8_h2_201214/resnet18_cifar100_ser_8_h2_201214_2014.pth'
W3='/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h3/resnet18_cifar100_ser_8_h3_201214/resnet18_cifar100_ser_8_h3_201214_2046.pth'
W4='/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h4/resnet18_cifar100_ser_8_h4_201214/resnet18_cifar100_ser_8_h4_201214_2057.pth'
W5='/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h5/resnet18_cifar100_ser_8_h5_201214/resnet18_cifar100_ser_8_h5_201214_2108.pth'
W6='/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h6/resnet18_cifar100_ser_8_h6_201214/resnet18_cifar100_ser_8_h6_201214_2119.pth'
W7='/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h7/resnet18_cifar100_ser_8_h7_201214/resnet18_cifar100_ser_8_h7_201214_2131.pth'
W8='/artic/t-kaneko/work/val_bit_serial/weights/resnet18_cifar100_ser_8_h8/resnet18_cifar100_ser_8_h8_201214/resnet18_cifar100_ser_8_h8_201214_2142.pth'

# W8='/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h8_200909_1644.pth'
# W7='/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h7_200909_1656.pth'
# W6='/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h6_200909_1707.pth'
# W5='/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h5_200909_1718.pth'
# W4='/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h4_200909_1730.pth'
# W3='/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h3_200909_1752.pth'
# W2='/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h2_200909_1533.pth'
# W1='/work/j-suzuki/weight/different_fn/resnet18_cifar100_ser8_9_h1_200904_1710.pth'
for i in `seq 8`
do
    num="W${i}"
    eval weight='$'${num}
    echo $weight
    python3 test.py -m resnet18 -d cifar100 -df True -w $weight -hb $i -ds pollux01
done

# python3 test.py -hb 8 --dataset cifar100 -m resnet18 --weights /artic/j-suzuki/work/github/pytorch-cifar100/weights/pretrained/resnet18_100_ser8_8.pth
# python3 test.py -hb 7 --dataset cifar100 -m resnet18 --weights /artic/j-suzuki/work/github/pytorch-cifar100/weights/pretrained/resnet18_100_ser8_8_h7.pth
# python3 test.py -hb 6 --dataset cifar100 -m resnet18 --weights /artic/j-suzuki/work/github/pytorch-cifar100/weights/pretrained/resnet18_100_ser8_8_h6.pth
# python3 test.py -hb 5 --dataset cifar100 -m resnet18 --weights /artic/j-suzuki/work/github/pytorch-cifar100/weights/pretrained/resnet18_100_ser8_8_h5.pth
# python3 test.py -hb 4 --dataset cifar100 -m resnet18 --weights /artic/j-suzuki/work/github/pytorch-cifar100/weights/pretrained/resnet18_100_ser8_8_h4.pth
# python3 test.py -hb 3 --dataset cifar100 -m resnet18 --weights /artic/j-suzuki/work/github/pytorch-cifar100/weights/pretrained/resnet18_100_ser8_8_h3.pth
# python3 test.py -hb 2 --dataset cifar100 -m resnet18 --weights /artic/j-suzuki/work/github/pytorch-cifar100/weights/pretrained/resnet18_100_ser8_8_h2.pth
# python3 test.py -hb 1 --dataset cifar100 -m resnet18 --weights /artic/j-suzuki/work/github/pytorch-cifar100/weights/pretrained/resnet18_100_ser8_8_h1.pth



# python3 test.py -w weights/vgg16_cifar100_ser_8_h1/vgg16_cifar100_ser_8_h1_201009/vgg16_cifar100_ser_8_h1_201009_0454.pth -hb 1
# python3 test.py -w weights/vgg16_cifar100_ser_8_h2/vgg16_cifar100_ser_8_h2_201009/vgg16_cifar100_ser_8_h2_201009_0348.pth -hb 2
# python3 test.py -w weights/vgg16_cifar100_ser_8_h3/vgg16_cifar100_ser_8_h3_201009/vgg16_cifar100_ser_8_h3_201009_0241.pth -hb 3
# python3 test.py -w weights/vgg16_cifar100_ser_8_h4/vgg16_cifar100_ser_8_h4_201009/vgg16_cifar100_ser_8_h4_201009_0133.pth -hb 4
# python3 test.py -w weights/vgg16_cifar100_ser_8_h5/vgg16_cifar100_ser_8_h5_201009/vgg16_cifar100_ser_8_h5_201009_0025.pth -hb 5
# python3 test.py -w weights/vgg16_cifar100_ser_`8_h6/vgg16_cifar100_ser_8_h6_201008/vgg16_cifar100_ser_8_h6_201008_2317.pth -hb 6
# python3 test.py -w weights/vgg16_cifar100_ser_8_h7/vgg16_cifar100_ser_8_h7_201008/vgg16_cifar100_ser_8_h7_201008_2209.pth -hb 7