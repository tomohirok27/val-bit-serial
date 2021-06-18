import torch.nn as nn
from .modules.layer_ops import LayerOps


class Vgg(nn.Module):
    def __init__(self,
                 layer_ops,
                 num_classes=10,
                 groups=1,
                 width_per_group=64):
        super(Vgg, self).__init__()

        conv = layer_ops.conv
        batchnorm2d = layer_ops.batchnorm2d
        relu = layer_ops.relu
        linear = layer_ops.linear

        self._0batchnorm2d = batchnorm2d
        #self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        # １層目
        self.conv1_1 = conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_conv1_1 = batchnorm2d(64)
        self.conv1_2 = conv(64, 64, kernel_size=3, stride=1, padding=1, bias = False)
        self.bn_conv1_2 = batchnorm2d(64)
        self.relu = relu()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ２層目
        self.conv2_1 = conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_conv2_1 = batchnorm2d(128)
        self.conv2_2 = conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_conv2_2 = batchnorm2d(128)

        # ３層目
        self.conv3_1 = conv(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_conv3_1 = batchnorm2d(256)
        self.conv3_2 = conv(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_conv3_2 = batchnorm2d(256)
        self.conv3_3 = conv(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_conv3_3 = batchnorm2d(256)

        # ４層目
        self.conv4_1 = conv(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_conv4_1 = batchnorm2d(512)
        self.conv4_2 = conv(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_conv4_2 = batchnorm2d(512)
        self.conv4_3 = conv(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_conv4_3 = batchnorm2d(512)

        # ５層目
        self.conv5_1 = conv(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_conv5_1 = batchnorm2d(512)
        self.conv5_2 = conv(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_conv5_2 = batchnorm2d(512)
        self.conv5_3 = conv(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_conv5_3 = batchnorm2d(512)

        # FC層
        self.fc1 = linear(512, 4096, bias=True)
        self.fc2 = linear(4096, 4096, bias=True)
        self.fc3 = linear(4096, num_classes, bias=True)
        # self.fc3 = nn.Linear(4096, num_classes, bias=True)

    def forward(self, x):
        # １層目
        # print(x.size())
        x = self.relu(self.bn_conv1_1(self.conv1_1(x)))
        x = self.maxpool(self.relu(self.bn_conv1_2(self.conv1_2(x))))

        # ２層目
        x = self.relu(self.bn_conv2_1(self.conv2_1(x)))
        x = self.maxpool(self.relu(self.bn_conv2_2(self.conv2_2(x))))

        # ３層目
        x = self.relu(self.bn_conv3_1(self.conv3_1(x)))
        x = self.relu(self.bn_conv3_2(self.conv3_2(x)))
        x = self.maxpool(self.relu(self.bn_conv3_3(self.conv3_3(x))))

        # ４層目
        x = self.relu(self.bn_conv4_1(self.conv4_1(x)))
        x = self.relu(self.bn_conv4_2(self.conv4_2(x)))
        x = self.maxpool(self.relu(self.bn_conv4_3(self.conv4_3(x))))

        # ５層目
        x = self.relu(self.bn_conv5_1(self.conv5_1(x)))
        x = self.relu(self.bn_conv5_2(self.conv5_2(x)))
        x = self.maxpool(self.relu(self.bn_conv5_3(self.conv5_3(x))))

        # FC層
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _vgg16(quant,num_classes, fixed_num,
            fract_num, high_bits, diff_fn, settings, **kwargs):
    layer_ops = LayerOps(settings, quant, fixed_num, fract_num,
                         high_bits, diff_fn)
    return Vgg(layer_ops, num_classes, **kwargs)

def vgg16(settings=None,
             quant='bit_serial',
             num_classes=10,
             fixed_num=8,
             fract_num=4,
             high_bits=None,
             diff_fn=None,
             **kwargs):

    return _vgg16(quant=quant,
                    num_classes=num_classes,
                    fixed_num=fixed_num,
                    fract_num=fract_num,
                    high_bits=high_bits,
                    diff_fn=diff_fn,
                    settings=settings,
                    **kwargs)


