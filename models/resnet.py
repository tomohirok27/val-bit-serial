import torch.nn as nn
from .modules.layer_ops import LayerOps


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, layer_ops, stride=1):
        super().__init__()

        # instance layer ops
        conv = layer_ops.conv
        batchnorm2d = layer_ops.batchnorm2d
        relu = layer_ops.relu

        # residual function
        self.residual_function = nn.Sequential(
            conv(in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=stride,
                 padding=1,
                 bias=False), batchnorm2d(out_channels), relu(inplace=True),
            conv(out_channels,
                 out_channels * BasicBlock.expansion,
                 kernel_size=3,
                 padding=1,
                 bias=False), batchnorm2d(out_channels * BasicBlock.expansion))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                conv(in_channels,
                     out_channels * BasicBlock.expansion,
                     kernel_size=1,
                     stride=stride,
                     bias=False),
                batchnorm2d(out_channels * BasicBlock.expansion))
        self.relu = relu(inplace=True)

    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, layer_ops, stride=1):
        super().__init__()

        # instance layer ops
        conv = layer_ops.conv
        batchnorm2d = layer_ops.batchnorm2d
        relu = layer_ops.relu

        self.residual_function = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=1, bias=False),
            batchnorm2d(out_channels),
            relu(inplace=True),
            conv(out_channels,
                 out_channels,
                 stride=stride,
                 kernel_size=3,
                 padding=1,
                 bias=False),
            batchnorm2d(out_channels),
            relu(inplace=True),
            conv(out_channels,
                 out_channels * BottleNeck.expansion,
                 kernel_size=1,
                 bias=False),
            batchnorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                conv(in_channels,
                     out_channels * BottleNeck.expansion,
                     stride=stride,
                     kernel_size=1,
                     bias=False),
                batchnorm2d(out_channels * BottleNeck.expansion))
        self.relu = relu(inplace=True)

    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_block, layer_ops, num_classes=100):
        super().__init__()

        self.in_channels = 64

        conv = layer_ops.conv
        fc = layer_ops.linear

        self.conv1 = nn.Sequential(
            conv(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 64, num_block[0], layer_ops, 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], layer_ops, 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], layer_ops, 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], layer_ops, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = fc(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, layer_ops, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_channels, out_channels, layer_ops, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def _resnet(block,
            layers,
            quant,
            num_classes,
            fixed_num,
            fract_num,
            high_bits,
            diff_fn,
            settings,
            th=0.,
            **kwargs):
    layer_ops = LayerOps(settings, quant, fixed_num, fract_num, high_bits,
                         diff_fn, th)
    return ResNet(block, layers, layer_ops, num_classes, **kwargs)


def resnet18(settings=None,
             quant='bit_serial',
             num_classes=10,
             fixed_num=8,
             fract_num=4,
             high_bits=None,
             diff_fn=None,
             th=0.,
             **kwargs):

    return _resnet(BasicBlock, [2, 2, 2, 2],
                   quant=quant,
                   num_classes=num_classes,
                   fixed_num=fixed_num,
                   fract_num=fract_num,
                   high_bits=high_bits,
                   diff_fn=diff_fn,
                   settings=settings,
                   th=th,
                   **kwargs)


def resnet34(settings=None,
             quant='bit_serial',
             num_classes=1000,
             fixed_num=8,
             fract_num=4,
             high_bits=None,
             diff_fn=None,
             **kwargs):

    return _resnet(BasicBlock, [3, 4, 6, 3],
                   quant=quant,
                   num_classes=num_classes,
                   fixed_num=fixed_num,
                   fract_num=fract_num,
                   high_bits=high_bits,
                   diff_fn=diff_fn,
                   settings=settings,
                   **kwargs)


def resnet50(settings=None,
             quant='bit_serial',
             num_classes=1000,
             fixed_num=8,
             fract_num=4,
             high_bits=None,
             diff_fn=None,
             **kwargs):

    return _resnet(BottleNeck, [3, 4, 6, 3],
                   quant=quant,
                   num_classes=num_classes,
                   fixed_num=fixed_num,
                   fract_num=fract_num,
                   high_bits=high_bits,
                   diff_fn=diff_fn,
                   settings=settings,
                   **kwargs)


def resnet101(settings=None,
              quant='bit_serial',
              num_classes=1000,
              fixed_num=8,
              fract_num=4,
              high_bits=None,
              diff_fn=None,
              **kwargs):

    return _resnet(BottleNeck, [3, 4, 23, 3],
                   quant=quant,
                   num_classes=num_classes,
                   fixed_num=fixed_num,
                   fract_num=fract_num,
                   high_bits=high_bits,
                   diff_fn=diff_fn,
                   settings=settings,
                   **kwargs)


def resnet152(settings=None,
              quant='bit_serial',
              num_classes=1000,
              fixed_num=8,
              fract_num=4,
              high_bits=None,
              diff_fn=None,
              **kwargs):

    return _resnet(BottleNeck, [3, 8, 36, 3],
                   quant=quant,
                   num_classes=num_classes,
                   fixed_num=fixed_num,
                   fract_num=fract_num,
                   high_bits=high_bits,
                   diff_fn=diff_fn,
                   settings=settings,
                   **kwargs)
