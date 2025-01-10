"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # the learnable weights of shape (in_features, out_features
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features, 
                                                   device=device, dtype=dtype, requires_grad=True))
        
        # the learnable bias of shape (out_features).
        self.bias = Parameter(
            ops.reshape(
                init.kaiming_uniform(self.out_features, 1, 
                                                     device=device, dtype=dtype, requires_grad=True),
                (1, self.out_features)
            )
        )
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        out = out + (ops.broadcast_to(self.bias, out.shape))
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        '''
        Takes in a tensor of shape (B,X_0,X_1,...), and flattens all non-batch dimensions 
        so that the output is of shape (B, X_0 * X_1 * ...)
        '''
        ### BEGIN YOUR SOLUTION
        ndim = len(X.shape)
        assert(ndim >= 2)
        dim_flat = 1
        for i in range(1, ndim):
            dim_flat *= X.shape[i]

        out = ops.reshape(X, (X.shape[0], dim_flat))
        return out
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n = len(self.modules)
        assert n > 0

        out = self.modules[0](x)
        if n == 1:
            return out
        
        for i in range(1, n):
            out = self.modules[i](out)
        
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # get onehot code for y
        num_entries, num_class = logits.shape
        y_onehot = init.one_hot(num_class, y, y.device, logits.dtype, False)

        # use onehot y to get logits[0:n, y]
        logits_y = ops.summation(logits * y_onehot, axes=(1,))

        # return loss = logsumexp(logits) - logits[y]
        loss = ops.logsumexp(logits, axes=(1,)) - logits_y

        num_entries = np.array(num_entries, dtype=logits.dtype)
        return ops.summation(loss) / num_entries
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    '''
    variables:
    - `weight` - the learnable weights of size `dim`, elements initialized to 1.
    - `bias` - the learnable bias of size `dim`, elements initialized to 0.
    - `running_mean` - the running mean used at evaluation time, elements initialized to 0.
    - `running_var` - the running (unbiased) variance used at evaluation time, elements initialized to 1. 
    '''
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.eps = Tensor(eps, device=device, dtype=dtype, requires_grad=False)

        # change: use float64 type running_mean (fail)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        self.gamma_old = Tensor(1.0 - momentum, device=device, dtype=dtype, requires_grad=False)
        self.gamma_new = Tensor(momentum, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # compute mean over the 
        assert len(x.shape) == 2

        # compute mean and variance in the batch
        mu = ops.mean(x, (0,))
        var = ops.variance(x, (0,))
        
        # update running mean and var use formula:
        # \hat{x_{new}} = (1 - m) \hat{x_{old}} + m x_{observed}
        # I close gradient when compute running mean (is it right?)     
        assert (not self.running_mean.requires_grad)

        # if is evaluating mode, use running mean and running variance
        # if not self.training:
        #     mu = self.running_mean
        #     var = self.running_var

        if self.training:
            # # compute normalize
            # # y = w  \frac{z_i - E[Z]}{((Var[Z]+\epsilon)^{1/2})} + b
            self.running_mean = ops.broadcast_to(self.gamma_old, mu.shape) *  self.running_mean + \
                ops.broadcast_to(self.gamma_new, mu.shape) *  mu.data
            self.running_var = ops.broadcast_to(self.gamma_old, var.shape) *  self.running_var + \
                ops.broadcast_to(self.gamma_new, var.shape) *  var.data   
            shape = x.shape
            mu = ops.broadcast_to(mu, shape)
            var = ops.broadcast_to(var, shape)
            w = ops.broadcast_to(self.weight, shape)
            b = ops.broadcast_to(self.bias, shape)
            eps = ops.broadcast_to(self.eps, shape)

            out = w * (
                (x - mu) / ops.sqrt(var + eps)
            ) + b

        else:
            shape = x.shape
            mu = ops.broadcast_to(self.running_mean, shape)
            var = ops.broadcast_to(self.running_var, shape)
            eps = ops.broadcast_to(self.eps, shape)
            w = ops.broadcast_to(self.weight, shape)
            b = ops.broadcast_to(self.bias, shape)
            out = (x - mu) / ops.sqrt(var + eps)
            out = w * out + b

        return out
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    '''
    - `weight` - the learnable weights of size `dim`, elements initialized to 1.
    - `bias` - the learnable bias of shape `dim`, elements initialized to 0.
    '''
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))

        self.eps = Tensor(eps, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    # assumption x is 2-dim
    def forward(self, x: Tensor) -> Tensor:
        # check dimension for x
        assert len(x.shape) == 2

        ### BEGIN YOUR SOLUTION
        mu = ops.mean(x, (1,), True)
        var = ops.variance(x, (1,), True)
        
        shape = x.shape
        mu = ops.broadcast_to(mu, shape)
        var = ops.broadcast_to(var, shape)
        w = ops.broadcast_to(self.weight, shape)
        b = ops.broadcast_to(self.bias, shape)
        eps = ops.broadcast_to(self.eps, shape)

        out = w * (
            (x - mu) / ops.sqrt(var + eps)
        ) + b

        return out
        ### END YOUR SOLUTION


class Dropout(Module):
    ''' randomly zeroes some of the elements of the input tensor with probability `p`
    Dropout only applied at train mode
    '''
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training: return x
        
        # generate random mask with prob 1-p being 1
        mask = init.randb(*(x.shape), p=1.0 - self.p, device=x.device, dtype=x.dtype)
        mask = mask / ops.broadcast_to(
            Tensor(1 - self.p, device=x.device, dtype=x.dtype, requires_grad=False),
            x.shape
        )

        # mask out input
        return mask * x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
