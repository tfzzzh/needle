"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # init kernel tensor of shape [K, K, Cin, Cout]
        kshape = (kernel_size, kernel_size, in_channels, out_channels)
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        self.weight = Parameter(
            init.kaiming_uniform(fan_in, fan_out, 
                                 device=device, dtype=dtype, shape=kshape, requires_grad=True)
            )

        # init bias of shape [Cout]
        self.use_bias = bias
        if self.use_bias:
            high = 1.0 / np.sqrt(in_channels * kernel_size * kernel_size)
            self.bias = Parameter(init.rand(out_channels, low=0.0, high=high, 
                                             device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        '''
        X: tensor of shape [N, C, H, W]
        out: tensor of shape [N, Cout, H', W']
        '''
        ### BEGIN YOUR SOLUTION
        # transform X from [N,C,H,W] to [N, H, W, C]
        x = ops.Permute((0,2,3,1))(x)
        
        # same padding
        assert self.kernel_size % 2 == 1
        width = self.kernel_size // 2

        # Y = Conv(X, W) + b
        out = ops.conv(x, self.weight, self.stride, padding=width)

        if self.use_bias:
            out = out + ops.broadcast_to(self.bias, out.shape)

        # transform Y from [N, H', W', C'] to [N, C', H', W']
        out = ops.Permute((0,3,1,2))(out)

        return out
        ### END YOUR SOLUTION