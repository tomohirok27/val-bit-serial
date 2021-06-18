import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class SerialBitF(autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, fn):
        th = 2**(num_bits - fn - 1)
        ctx.save_for_backward(input, torch.tensor(th).to(input.device.type))

        # x to fixed
        x = (input * (2**fn)).int()
        max_value_int = 2**(num_bits - 1)
        x = x.clamp(-max_value_int, max_value_int)
        out = x.float() / (2**fn)

        # like serial bit
        add_value = torch.ones_like(out) * 2**(-(fn + 1))
        out = torch.where(out < 0, out + add_value, out)
        out = torch.where(out > 0, out - add_value, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        input, th = ctx.saved_tensors
        out = torch.where((input >= -th.item()) & (input <= th.item()),
                          grad_out, torch.zeros_like(grad_out))
        return out, None, None


def increment_weight(weight, num_bits=8, fn=6, high_bits=8):
    max_log2 = (num_bits - 1) - fn - 1
    max_value = torch.ones_like(weight) * 2**max_log2
    out = torch.where(weight > 0, max_value, weight)
    out = torch.where(weight < 0, -max_value, out)

    for i in range(high_bits - 1):
        diff = weight - out
        add_value = torch.ones_like(weight) * 2**(max_log2 - (i + 1))
        add_tensor = torch.zeros_like(weight)
        add_tensor = torch.where(diff > 0, add_value, add_tensor)
        add_tensor = torch.where(diff < 0, -add_value, add_tensor)
        out = out + add_tensor

    return out


# serial_bit = PeseudSBF.apply
serial_bit = SerialBitF.apply


class SBConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 fixed_num=8,
                 fract_num=6,
                 high_bits=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.fixed_num = fixed_num
        self.fract_num = fract_num
        self.high_bits = high_bits

        # serial quantization param
        if high_bits:
            fract_serial = fract_num - (fixed_num - high_bits)
        self.num_bits = high_bits if high_bits else fixed_num
        self.fract_bits = fract_serial if high_bits else fract_num
        self.serial_rshift = fixed_num - high_bits if high_bits else 0

        self.init = True

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        if self.high_bits:
            # initialize weight
            if self.init:
                self.fixed_weight = serial_bit(self.weight, self.fixed_num,
                                               self.fract_num)
                self.fixed_weight = increment_weight(self.fixed_weight,
                                                     self.fixed_num,
                                                     self.fract_num,
                                                     self.high_bits)
                self.weight = torch.nn.Parameter(self.fixed_weight,
                                                 requires_grad=False)
                self.init = False
            # return self.conv2d_forward(input, self.weight)
            return self._conv_forward(input, self.weight)                

        else:
            self.fixed_weight = serial_bit(self.weight, self.num_bits,
                                           self.fract_bits)

            # return self.conv2d_forward(input, self.fixed_weight)
            return self._conv_forward(input, self.fixed_weight)        


class SBLinear(nn.Linear):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 fixed_num=16,
                 fract_num=7,
                 high_bits=None):

        super(SBLinear, self).__init__(in_channels, out_channels)
        self.incin_channels = in_channels
        self.outout_channels = out_channels
        self.weight = Parameter(torch.Tensor(out_channels, in_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.fixed_num = fixed_num
        self.fract_num = fract_num
        self.high_bits = high_bits

        if high_bits:
            fract_serial = fract_num - (fixed_num - high_bits)
        self.num_bits = high_bits if high_bits else fixed_num
        self.fract_bits = fract_serial if high_bits else fract_num
        self.serial_rshift = fixed_num - high_bits if high_bits else 0

        self.init = True

    def forward(self, input):
        if self.high_bits:
            if self.init:
                self.fixed_weight = serial_bit(self.weight, self.fixed_num,
                                               self.fract_num)
                self.fixed_weight = increment_weight(self.fixed_weight,
                                                     self.fixed_num,
                                                     self.fract_num,
                                                     self.high_bits)
                self.weight = torch.nn.Parameter(self.fixed_weight,
                                                 requires_grad=False)
                self.init = False
            return F.linear(input, self.fixed_weight, self.bias)

        else:
            self.fixed_weight = serial_bit(self.weight, self.fixed_num,
                                           self.fract_num)
            return F.linear(input, self.fixed_weight, self.bias)


class PseudoSBActF(autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, fn, is_relu=False):
        ctx.save_for_backward(input)
        # x to fixed
        a = (input * (2**fn)).int()
        max_value_int = 2**(num_bits - 1) - 1
        min_value = 0 if is_relu else -max_value_int
        a = a.clamp(min_value, max_value_int) & (2**num_bits - 1)
        # fixed to x
        x = a & (2**num_bits - 1)
        y = torch.where((x >> (num_bits - 1)) != 0, -((2**num_bits) - x),
                        x & (2**(num_bits - 1) - 1))
        out = y.float() / (2**fn)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input, None, None, None


sb_act = PseudoSBActF.apply


class SBReLU(nn.Module):
    def __init__(self, num_bits=8, fn=4):
        super(SBReLU, self).__init__()
        self.num_bits = num_bits
        self.fn = fn

    def forward(self, x):
        relu_out = sb_act(x, self.num_bits, self.fn, True)
        return relu_out


class SBBN2d(nn.BatchNorm2d):
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 num_bits=8,
                 fn=6):
        super(SBBN2d, self).__init__(num_features, eps, momentum, affine,
                                     track_running_stats)
        self.num_bits = num_bits
        self.fn = fn

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # quantization weight
        self.fixed_weight = serial_bit(self.weight, self.num_bits, self.fn)
        # quantization bias
        self.fixed_bias = serial_bit(self.bias, self.num_bits, self.fn)

        return F.batch_norm(input, self.running_mean, self.running_var,
                            self.fixed_weight, self.fixed_bias, self.training
                            or not self.track_running_stats,
                            exponential_average_factor, self.eps)
