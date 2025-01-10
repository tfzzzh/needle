from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        '''
        Applies a numerically stable logsoftmax function to the input by subtracting off the maximum elements. 
        Assume the input NDArray is 2 dimensional and we are doing softmax over `axis=1`.

        {LogSoftmax}(z) = log ({exp(z_i - max z)}/{sum_{i} exp(z_i - max z)})
        '''
        ### BEGIN YOUR SOLUTION
        assert (len(Z.shape) == 2)
        z_max = array_api.max(Z, axis=1, keepdims=True)
        z = Z - z_max
        z_exp = array_api.exp(z)
        z_sum = array_api.sum(z_exp, axis=1, keepdims=True)
        z_sum = array_api.log(z_sum)
    
        return  z - z_sum
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        '''
        gradZ = out_grad - sum(out_grad, 1) * softmax(Z)
        '''
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        Z_smx = stable_smt(Z, axes=(1,))

        out_grad_s = summation(out_grad, axes=(1,), keep_dim=True)
        out_grad_s = broadcast_to(out_grad_s, Z.shape)

        return out_grad - (out_grad_s * Z_smx)
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


def stable_smt(Z: Tensor, axes: Tuple[int]) -> Tensor:
    m = array_api.max(Z.realize_cached_data(), axis=axes, keepdims=True)
    m = array_api.broadcast_to(m, Z.shape)
    A = exp(Z - m)
    Anorm = summation(A, axes, keep_dim=True)
    return A / broadcast_to(Anorm, A.shape)


class LogSumExp(TensorOp):
    '''
    use stablized formulation: 
    LogSumExp(z) = log (sum_{i} exp (z_i - max{z})) + max{z}
    '''
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # assumption: axes not out of range
        # compute max for each row
        z_max = array_api.max(Z, axis=self.axes, keepdims=True)

        # compute exp with max removed
        z_exp = array_api.exp(Z - z_max)

        # compute log sum of exp over the axes
        z_sum = array_api.log(array_api.sum(z_exp, axis=self.axes))

        return z_sum + array_api.reshape(z_max, z_sum.shape)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # compute stable soft max for the node
        Z = node.inputs[0]
        # m = array_api.max(Z.realize_cached_data(), axis=self.axes, keepdims=True)
        # m = array_api.broadcast_to(m, Z.shape)
        # A = exp(Z - m)
        # Anorm = summation(A, self.axes, keep_dim=True)
        # A = A / broadcast_to(Anorm, A.shape)
        A = stable_smt(Z, self.axes)
        
        # broad_cast out_grad to Z
        shape_kd = recover_keepdim(Z.shape, self.axes)
        out_grad = reshape(out_grad, shape_kd)
        out_grad = broadcast_to(out_grad, Z.shape)

        # return gradient
        return out_grad * A
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

