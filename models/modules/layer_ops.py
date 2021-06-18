import torch.nn as nn
from .quant_fixed import FixedConv2d, FixedLinear, FixedReLU, FixedBN2d
from .serial_bit import SBConv2d, SBLinear, SBReLU, SBBN2d


class LayerOps():
    def __init__(self,
                 settings,
                 quant='bit_serial',
                 fixed_num=8,
                 fract_num=9,
                 high_bits=None,
                 diff_fn=None,
                 th=0.):

        self.fixed_num = fixed_num
        self.fract_num = fract_num
        self.high_bits = high_bits

        # quantization flags
        self.is_bit_serial = True if quant == 'bit_serial' else False
        self.is_fixed = True if quant == 'fixed' else False

        self.diff_fn = diff_fn
        self.settings = settings
        self.conv_cnt = 0
        self.fc_cnt = 0

        self.th = th

    def conv(self,
             in_channels,
             out_channels,
             kernel_size,
             stride=1,
             padding=0,
             dilation=1,
             groups=1,
             bias=True,
             padding_mode='zeros'):

        if self.is_fixed | self.is_bit_serial:
            if self.is_fixed:
                conv_fn = FixedConv2d

            if self.is_bit_serial:
                conv_fn = SBConv2d
                # conv_fn = FixedConv2d

            fract_num = self.settings.CONV_FN_LIST[
                self.conv_cnt] if self.diff_fn else self.fract_num
            self.conv_cnt += 1

            return conv_fn(in_channels,
                           out_channels,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           groups=groups,
                           bias=bias,
                           padding_mode=padding_mode,
                           dilation=dilation,
                           fixed_num=self.fixed_num,
                           fract_num=fract_num,
                           high_bits=self.high_bits)

        else:
            return nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding,
                             groups=groups,
                             bias=bias,
                             padding_mode=padding_mode,
                             dilation=dilation)

    def linear(self, in_channels, out_channels, bias=True):

        if self.is_fixed | self.is_bit_serial:
            if self.is_fixed:
                linear_fn = FixedLinear
            if self.is_bit_serial:
                linear_fn = SBLinear
                # linear_fn = FixedLinear

            fract_num = self.settings.FC_FN_LIST[
                self.fc_cnt] if self.diff_fn else self.fract_num
            self.fc_cnt += 1

            return linear_fn(in_channels,
                             out_channels,
                             bias=bias,
                             fixed_num=self.fixed_num,
                             fract_num=fract_num,
                             high_bits=self.high_bits)
        else:
            return nn.Linear(in_channels, out_channels, bias=bias)

    def relu(self, inplace=False, num_bits=8, fn=4):

        if self.is_fixed | self.is_bit_serial:
            if self.is_fixed:
                return FixedReLU(num_bits=num_bits, fn=fn, th=self.th)
            if self.is_bit_serial:
                # return SBReLU(num_bits, fn)
                return FixedReLU(num_bits=num_bits, fn=fn, th=self.th)
        else:
            return nn.ReLU(inplace=inplace)

    def batchnorm2d(self,
                    num_features,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True):

        if self.is_fixed | self.is_bit_serial:
            if self.is_fixed:
                bn_fn = FixedBN2d
            if self.is_bit_serial:
                # bn_fn = SBBN2d
                bn_fn = FixedBN2d
            return bn_fn(num_features,
                         eps,
                         momentum,
                         affine,
                         track_running_stats,
                         num_bits=8,
                         fn=6)

        else:
            return nn.BatchNorm2d(num_features, eps, momentum, affine,
                                  track_running_stats)
