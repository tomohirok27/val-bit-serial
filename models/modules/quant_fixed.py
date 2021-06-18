import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


def corrcoef_torch_to_numpy(a, b):
    import numpy as np
    a = a.cpu().detach().numpy().flatten()
    b = b.cpu().detach().numpy().flatten()
    c = np.corrcoef(a, b)
    return c[0, 1]


class X_to_FixedF(autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, fn):
        ctx.save_for_backward(input)
        a = (input * (2**fn)).int()
        max_value_int = 2**(num_bits - 1)
        a = a.clamp(-max_value_int, max_value_int - 1)
        out = a.float() / (2**fn)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        input = ctx.saved_tensors[0]
        th = 0.5
        out = torch.where((input >= -th) & (input <= th), grad_out,
                          torch.zeros_like(grad_out))
        return out, None, None


# apply function
# x_to_fixed = FixedF.apply
x_to_fixed = X_to_FixedF.apply


class FixedConv2d(nn.Conv2d):
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

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        self.fixed_weight = x_to_fixed(self.weight, self.fixed_num,
                                       self.fract_num)

        return self.conv2d_forward(input, self.fixed_weight)


class FixedLinear(nn.Linear):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 fixed_num=16,
                 fract_num=7,
                 high_bits=None):
        super(FixedLinear, self).__init__(in_channels, out_channels)
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

    def forward(self, input):
        # weight quantization
        self.fixed_weight = x_to_fixed(self.weight, self.fixed_num,
                                       self.fract_num)
        self.fixed_bias = x_to_fixed(self.bias, self.fixed_num,
                                     self.fixed_num + 3)
        # self.fixed_weight = fixed_to_x(self.fixed_weight_bin, self.num_bits,
        #                                self.fract_bits, self.serial_rshift)
        # activation quantization
        out = F.linear(input, self.fixed_weight, self.fixed_bias)
        out = fixed_act(out, 8, 3)
        # out = fixed_to_x(out, 8, 3, 0)
        return out


class FixedActF(autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, fn, is_relu=False):
        ctx.save_for_backward(input)
        # x to fixed
        a = (input * (2**fn)).int()
        max_value_int = 2**(num_bits - 1) - 1
        min_value = 0 if is_relu else -max_value_int - 1
        a = a.clamp(min_value, max_value_int)
        out = a.float() / (2**fn)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input, None, None, None


fixed_act = FixedActF.apply


class FixedReLU(nn.Module):
    def __init__(self, num_bits=8, fn=4, th=0.):
        super(FixedReLU, self).__init__()
        self.num_bits = num_bits
        self.fn = fn

        self.th = th

    def forward(self, x):
        # # developing
        # x = act_predict(x, self.th)
        # act_save(x)
        relu_out = fixed_act(x, self.num_bits, self.fn, True)
        return relu_out


class FixedBN2d(nn.BatchNorm2d):
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 num_bits=8,
                 fn=6):
        super(FixedBN2d, self).__init__(num_features, eps, momentum, affine,
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
        self.fixed_weight = x_to_fixed(self.weight, self.num_bits, self.fn)
        # self.fixed_weight = fixed_to_x(self.fixed_weight_bin, self.num_bits,
        #                                self.fn, 0)
        # quantization bias
        self.fixed_bias = x_to_fixed(self.bias, self.num_bits, self.fn)
        # self.fixed_bias = fixed_to_x(self.fixed_bias_bin, self.num_bits,
        #                              self.fn, 0)

        return F.batch_norm(input, self.running_mean, self.running_var,
                            self.fixed_weight, self.fixed_bias, self.training
                            or not self.track_running_stats,
                            exponential_average_factor, self.eps)
