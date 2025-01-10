import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ''' 
    x ~ U[-a, a] with a = gain * sqrt(6 / (fan_in + fan_out))
    Note that only float32 version is implemented
    '''
    ### BEGIN YOUR SOLUTION
    device = kwargs['device'] if 'device' in kwargs else None
    dtype = kwargs['dtype'] if 'dtype' in kwargs else "float32"
    requires_grad = kwargs['requires_grad'] if 'requires_grad' in kwargs else False

    a = math.sqrt(6 / (fan_in + fan_out)) * gain
    x = rand(fan_in, fan_out, low=-a, high=a, device=device, dtype=dtype, requires_grad=requires_grad)
    return x
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    device = kwargs['device'] if 'device' in kwargs else None
    dtype = kwargs['dtype'] if 'dtype' in kwargs else "float32"
    requires_grad = kwargs['requires_grad'] if 'requires_grad' in kwargs else False

    a = math.sqrt(2 / (fan_in + fan_out)) * gain
    x = randn(fan_in, fan_out, std=a, device=device, dtype=dtype, requires_grad=requires_grad)
    return x
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    device = kwargs['device'] if 'device' in kwargs else None
    dtype = kwargs['dtype'] if 'dtype' in kwargs else "float32"
    requires_grad = kwargs['requires_grad'] if 'requires_grad' in kwargs else False

    a = math.sqrt(6 / fan_in)
    x = rand(fan_in, fan_out, low=-a, high=a, device=device, dtype=dtype, requires_grad=requires_grad)
    return x
    ### END YOUR SOLUTION



def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    device = kwargs['device'] if 'device' in kwargs else None
    dtype = kwargs['dtype'] if 'dtype' in kwargs else "float32"
    requires_grad = kwargs['requires_grad'] if 'requires_grad' in kwargs else False

    a = math.sqrt(2 / fan_in)
    x = randn(fan_in, fan_out, std=a, device=device, dtype=dtype, requires_grad=requires_grad)
    return x
    ### END YOUR SOLUTION