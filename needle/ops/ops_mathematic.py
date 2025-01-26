"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


def pyscalar2array(scalar: Union[int, float], reference_arr: Union[NDArray, Tensor]):
    if BACKEND == 'np':
        return array_api.array(scalar, dtype=reference_arr.dtype)
    
    elif BACKEND == 'nd':
        return array_api.array(scalar, dtype=reference_arr.dtype, device=reference_arr.device)
    
    else:
        raise NotImplementedError


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return mul_scalar(out_grad, self.scalar)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs[0], node.inputs[1]
        GA = multiply(divide(node, A), B)
        GB = multiply(node, log(A))

        return multiply(out_grad, GA), multiply(out_grad, GB) 
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: Union[int, float]):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        scalar = pyscalar2array(self.scalar, a)
        return array_api.power(a, scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if BACKEND == 'np':
            scalar = pyscalar2array(self.scalar, out_grad)
        else:
            scalar = self.scalar
        return out_grad * mul_scalar(power_scalar(node.inputs[0], scalar-1), scalar)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)

def sqrt(a: Tensor):
    return power_scalar(a, scalar=0.5)

class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad_a = out_grad / node.inputs[1]
        grad_b = out_grad * negate(
            node.inputs[0] / power_scalar(node.inputs[1], 2)
        )
        return (grad_a, grad_b)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar: Union[float, int, NDArray]):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if BACKEND == 'np':
            scalar = pyscalar2array(self.scalar, a)
            scalar = array_api.broadcast_to(scalar, a.shape)
        else:
            scalar = self.scalar

        return a / scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if BACKEND == 'np':
            scalar = pyscalar2array(self.scalar, node.realize_cached_data())
            scalar = array_api.broadcast_to(scalar, node.shape)
        else:
            scalar = self.scalar

        return  divide_scalar(out_grad, scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        assert(len(a.shape) >= 2)
        if self.axes is not None:
            assert(len(self.axes) == 2)
            i = self.axes[0]
            j = self.axes[1]
        else:
            i = -1
            j = -2

        perm = list(range(len(a.shape)))
        (perm[i], perm[j]) = (perm[j], perm[i])

        out = array_api.transpose(a, tuple(perm))
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)

class Permute(TensorOp):
    def __init__(self, axis: Tuple[int]):
        self.axis = axis
        self.reverse_perm = tuple(numpy.argsort(axis))

    def compute(self, a: NDArray):
        assert (len(self.axis) == a.ndim)
        return array_api.transpose(a, self.axis)
    
    def gradient(self, out_grad, node):
        return Permute(self.reverse_perm)(out_grad)

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if BACKEND == 'nd':
            a = a.compact()

        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape_old = node.inputs[0].shape
        return reshape(out_grad, shape_old)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


def get_broadcasted_axes(shape_old: Tuple[int], shape_new: Tuple[int]) -> Tuple[int]:
    """Determine which axes in shape_new are broadcasted from shape_old.

        Parameters:
        -----------
        shape_old : Tuple
            The original shape.
        shape_new : Tuple
            The resulting shape after broadcasting.
        Returns:
        --------
        Tuple[int]
            A tuple of axes (indices) in 'shape_new' that are broadcasted from 'shape_old'.
        Raises:
        -------
        AssertionError
            If a dimension mismatch is found (other than the case where shape_old has size 1).
    """
    dim, dim_old = len(shape_new), len(shape_old)
    axes = []

    for i in range(dim-1, -1, -1):
        j = i - dim + dim_old

        if j < 0:
            axes.append(i)
        
        else:
            sz_new, sz_old= shape_new[i], shape_old[j]
            if sz_new != sz_old:
                assert(sz_old == 1)
                axes.append(i)

    return tuple(reversed(axes))


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape
        self.shape_old = None

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.shape_old = a.shape
        self.axes = get_broadcasted_axes(shape_old=self.shape_old, shape_new=self.shape)
        return array_api.broadcast_to(a, self.shape) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if len(self.axes) != 0:
            grad = summation(out_grad, axes=self.axes)
        else:
            grad = out_grad
        grad = reshape(grad, self.shape_old)
        return grad
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


def recover_keepdim(shape_old: Tuple[int], axes: Union[Tuple[int], None]) -> Tuple[int]:
    shape = list(shape_old)

    if axes is not None:
        for axis in axes:
            shape[axis] = 1
    else:
        for axis in range(len(shape)):
            shape[axis] = 1
    
    return tuple(shape)


class Summation(TensorOp):
    def __init__(self, axes: Union[Tuple, None, int] = None):
        if axes is None or isinstance(axes, Tuple):
            self.axes = axes

        else:
            self.axes = (axes, )

        self.shape_old = None
        self.keep_dim = None


    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.shape_old = a.shape
        self.keep_dim = recover_keepdim(self.shape_old, self.axes)
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad = reshape(out_grad, self.keep_dim)
        grad = broadcast_to(grad, self.shape_old)
        return grad
        ### END YOUR SOLUTION


def summation(a, axes=None, keep_dim=False):
    """
    Perform summation over specified axes of the input tensor.

    Parameters:
    a (Tensor): The input tensor to be summed.
    axes (int, tuple of int, or None): The axes along which to sum. If None, sum over all axes.
    keep_dim (bool): If True, retains reduced dimensions with length 1. Default is False.

    Returns:
    Tensor: The result of the summation, with dimensions reduced according to the axes specified.
    """
    if  not keep_dim:
        return Summation(axes)(a)
    else:
        shape_kd = recover_keepdim(a.shape, axes)
        out = Summation(axes)(a)
        out = reshape(out, shape_kd)
        return out


def mean(a, axes=None, keep_dim=False):
    if axes is None:
        axes = tuple(range(len(a.shape)))
    
    n = 1
    for axis in axes:
        n *= a.shape[axis]
    
    a_sum = summation(a, axes, keep_dim)
    n = pyscalar2array(n, a)
    n = array_api.broadcast_to(n, a_sum.shape)
    return a_sum / n


def variance(a, axes=None, keep_dim=False):
    '''
    use formula variance = E((x - E(x))**2)
    '''
    Ex = mean(a, axes, keep_dim)
    x = a - broadcast_to(Ex, a.shape)
    return mean(x * x, axes, keep_dim)


# get dimension of an NDarray
def get_ndim(a):
    return len(a.shape)

def get_matmul_broadcast_axes(shape1: Tuple, shape2: Tuple):
    '''
    suppose shape1 and shape2 are shapes of two NDArray which are input of np.matmul
    return the shape of these matrix after broadcast along with broadcast axes 

    shape1: len(shape1) >= 2
    shape2: len(shape2) >= 2
    '''
    dim1, dim2 = len(shape1), len(shape2)
    if (dim1 == 2 and dim2 == 2):
        return shape1, (), shape2, ()
    
    dim_br = max(dim1, dim2)
    shape1_br = [0] * dim_br
    axes1 = []
    shape2_br = [0] * dim_br
    axes2 = []

    shape1_br[-1], shape1_br[-2] = shape1[-1], shape1[-2]
    shape2_br[-1], shape2_br[-2] = shape2[-1], shape2[-2]

    for i in range(dim_br-3, -1, -1):

        i1 = i - dim_br + dim1
        i2 = i - dim_br + dim2

        if (i1 < 0):
            assert(i2 >= 0)
            axes1.append(i)
            shape1_br[i] = shape2[i2]
            shape2_br[i] = shape2[i2]

        elif (i2 < 0):
            assert(i1 >= 0)
            axes2.append(i)
            shape1_br[i] = shape1[i1]
            shape2_br[i] = shape1[i1]
        
        else:
            s1, s2 = shape1[i1], shape2[i2]
            if (s1 == s2):
                shape1_br[i] = shape1[i1]
                shape2_br[i] = shape2[i2]

            else:
                if (s1 == 1):
                    shape1_br[i] = shape2[i2]
                    shape2_br[i] = shape2[i2]
                    axes1.append(i)
                else:
                    assert(s2 == 1)
                    shape1_br[i] = shape1[i1]
                    shape2_br[i] = shape1[i1]
                    axes2.append(i)

    return tuple(shape1_br), tuple(reversed(axes1)), tuple(shape2_br), tuple(reversed(axes2))  


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # grad_a = matmul(out_grad, transpose(node.inputs[1]))
        # grad_b = matmul(transpose(node.inputs[0]), out_grad)
        # return grad_a, grad_b
        A = node.inputs[0]
        B = node.inputs[1]
        sa, sb = A.shape, B.shape

        # special case: 1D
        if len(sa) == 1:
            A = reshape(A, (1, sa[0]))
        
        if len(sb) == 1:
            B = reshape(B, (sb[0], 1))

        GA, GB = self._gradient_mat(out_grad, A, B)

        GA = reshape(GA, sa)
        GB = reshape(GB, sb)

        return GA, GB
        ### END YOUR SOLUTION

    def _gradient_mat(self, adjoin, A, B):
        ''' gradient when inputs are matrix
        '''
        shapeA = A.shape
        shapeB = B.shape

        # broadcast A, B
        shapeAbr, axesA, shapeBbr, axesB = get_matmul_broadcast_axes(shapeA, shapeB)

        if (len(axesA) != 0):
            A = broadcast_to(A, shapeAbr)
        
        if (len(axesB) != 0):
            B = broadcast_to(B, shapeBbr)
        
        # compute graident to A
        GA = matmul(adjoin, transpose(B))
        GB = matmul(transpose(A), adjoin)

        # reduction and reshape
        if (len(axesA) != 0):
            GA = summation(GA, axesA)

        if (len(axesB) != 0):
            GB = summation(GB, axesB)

        return GA, GB


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X = node.inputs[0]
        return multiply(out_grad, power_scalar(X, -1))
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad, node)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a * (a >= 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        xarr = node.inputs[0].realize_cached_data()
        # yarr = array_api.ones_like(xarr)
        if BACKEND == 'np':
            yarr = array_api.ones_like(xarr)
        
        else:
            assert BACKEND == 'nd'
            yarr = array_api.array(numpy.ones_like(xarr), dtype=xarr.dtype, device=xarr.device)

        # yarr[xarr <= 0] = 0
        yarr = yarr * (xarr > 0)
        return mul_scalar(out_grad, yarr)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        '''
        D tanh = 1 - (tanh)^2
        '''
        ### BEGIN YOUR SOLUTION
        one_arr = pyscalar2array(1.0, node)

        if BACKEND == 'nd':
            one_arr = array_api.broadcast_to(one_arr, node.shape)
            one_arr = one_arr.compact()

        g = add_scalar(-power_scalar(node, 2.0), one_arr)
        return out_grad * g
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(args) > 0
        out = array_api.stack(args, axis=self.axis)

        # reshape to pytorch's flavor
        n = len(args)
        shape = list(args[0].shape)
        # if len(shape) == self.axis: shape.append(n) 
        shape.insert(self.axis, n)
        out = array_api.reshape(out, tuple(shape))

        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # shape = node.inputs[0][0].shape
        # dim_input = len(shape)
        # if self.axis < dim_input:
        #     chunk_size = shape[self.axis]
        #     num_chunk = node.shape[self.axis] // chunk_size
        #     return split(out_grad, num_chunk, self.axis)

        # else:
        #     assert self.axis == dim_input
        #     chunk_size = 1
        #     num_chunk = node.shape[self.axis] // chunk_size
        #     grad_tuple = split(out_grad, num_chunk, self.axis)
        #     grad_tuple = grad_tuple.tuple()

        #     out = []
        #     for grad in grad_tuple:
        #         out.append(reshape(grad, shape))

        #     out = make_tuple(*out)
        #     return out
        
        # grad_tuple = split(out_grad, self.axis)
        # grad_tuple = grad_tuple.tuple()

        # shape = node.inputs[0][0].shape
        # out = []
        # for grad in grad_tuple:
        #     out.append(reshape(grad, shape))

        # out = make_tuple(*out)
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis
        # self.num_chunk = num_chunk

    def compute(self, A):
        assert A.ndim >= 2, f"A.ndim == {A.ndim}"
        ### BEGIN YOUR SOLUTION
        # convert to list of (...1...)
        num_chunk = A.shape[self.axis]
        Asubs = array_api.split(A, num_chunk, axis=self.axis)

        # remove self.axis
        shape_sub = tuple(A.shape[i] for i in range(A.ndim) if i != self.axis)
        for i in range(len(Asubs)):
            Asubs[i] = array_api.reshape(Asubs[i].compact(), shape_sub)

        return tuple(Asubs)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        '''
        parameters:
        ----------------------
        axes: tuple
            axes to apply dilation

        dilation: int
             the dilation amount 
        '''
        assert isinstance(axes,tuple)
        assert isinstance(dilation, int)
        self.axes = axes
        self.dilation = dilation
        self.stride = dilation + 1

    def compute(self, a: NDArray):
        """
        Dilate the input NDArray along specified axes by a given dilation factor.

        Parameters
        ----------
        a : NDArray
            The input array to be dilated.

        Returns
        -------
        NDArray
            A new NDArray with the specified dilation.
        """
        ### BEGIN YOUR SOLUTION
        assert self.stride >= 1
        if (len(self.axes) == 0): return a
        if (self.stride == 1): return a

        # for an axis with size sz, after stride -> sz * stride
        # create an array large enough to store the array
        shape_new = list(a.shape)
        for dim in self.axes:
            shape_new[dim] *= self.stride
        shape_new = tuple(shape_new)
        out = array_api.array(
            numpy.zeros(shape=shape_new, dtype=a.dtype),
            dtype = a.dtype,
            device = a.device
        )

        # for an dilated axis, make the point at i -> i * stride
        indices = [slice(0, shape_new[i]) for i in range(a.ndim)]
        for dim in self.axes:
            indices[dim] = slice(0, shape_new[dim], self.stride)

        out[tuple(indices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        """
        # The gradient of the dilate operation is the undilate operation, which reverses the dilation.
        return undilate(out_grad, self.axes, self.dilation)
        
        Parameters:
        out_grad : Tensor
            The gradient of the output tensor.

        Returns:
        Tensor
            The gradient of the input tensor.
        """
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation
        self.stride = dilation + 1

    def compute(self, a):
        """
        Undilate the input NDArray along specified axes by a given dilation factor.

        Parameters
        ----------
        a : NDArray
            The input array to be undilated.

        Returns
        -------
        NDArray
            A new NDArray with the specified undilation.
        """
        ### BEGIN YOUR SOLUTION
        assert self.stride >= 1
        if (len(self.axes) == 0): return a
        if (self.stride == 1): return a

        # allocate output memory
        shape_new = list(a.shape)
        for dim in self.axes:
            assert shape_new[dim] % self.stride == 0
            shape_new[dim] = shape_new[dim] // self.stride

        shape_new = tuple(shape_new)
        out = array_api.array(
            numpy.zeros(shape=shape_new, dtype=a.dtype),
            dtype = a.dtype,
            device = a.device
        )

        # get relevant indices
        indices = [slice(0, a.shape[i]) for i in range(a.ndim)]
        for dim in self.axes:
            indices[dim] = slice(0, a.shape[dim], self.stride)

        indices_out = tuple(slice(0, shape_new[i]) for i in range(a.ndim))
        out[indices_out] = a[tuple(indices)]
        return out

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)

class PadZero(TensorOp):
    def __init__(self, pad_width: Tuple[Tuple[int]]):
        self.pad_width = pad_width

    def compute(self, a: NDArray) -> NDArray:
        assert len(self.pad_width) == a.ndim
        if all(width == (0, 0) for width in self.pad_width):
            return a
        
        return array_api.pad(a, self.pad_width)

    def gradient(self, out_grad, node):
        return unpad_zero(out_grad, self.pad_width)
    
def pad_zero(a: Tensor, pad_width: Tuple[Tuple[int]]) -> Tensor:
    return PadZero(pad_width)(a)

class UnPadZero(TensorOp):
    def __init__(self, pad_width: Tuple[Tuple[int]]):
        self.pad_width = pad_width

    def compute(self, a: NDArray) -> NDArray:
        assert len(self.pad_width) == a.ndim
        if all(width == (0, 0) for width in self.pad_width):
            return a
        
        indices = []
        for i in range(a.ndim):
            start = self.pad_width[i][0]
            end = a.shape[i] - self.pad_width[i][1]
            assert start < end
            indices.append(slice(start, end))

        return a[tuple(indices)]
    
    def gradient(self, out_grad, node):
        return pad_zero(out_grad, self.pad_width)
    
def unpad_zero(a: Tensor, pad_width: Tuple[Tuple[int]]) -> Tensor:
    return UnPadZero(pad_width)(a)

class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride if stride is not None else 1
        self.padding = padding if padding is not None else 0
        self.pad_width = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0,0))

    def compute(self, A: NDArray, B: NDArray):
        """
        Perform a 2D convolution operation on the input NDArray using the given kernel NDArray.

        Parameters
        ----------
        A : NDArray, shape: [N, H, W, Cin]
            The input array to be convolved

        B : NDArray, shape: [K, K, Cin, Cout]

        Returns
        -------
        NDArray, shape [N, Ho, Wo, Cout]
        """
        ### BEGIN YOUR SOLUTION
        # check dims of A and B
        assert A.ndim == 4 and B.ndim == 4

        # check shape compatible for A and B
        assert A.shape[3] == B.shape[2]

        # when padding >= 1, pad axes [1, 2]
        if self.padding >= 1:
            pad_width = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0,0))
            A = array_api.pad(A, pad_width)

        # allocate an array to store output. finally size [N, Ho, Wo, Cout]
        N, H, W, Cin = A.shape
        K1, K2, _, Cout = B.shape
        Ho_full, Wo_full = H - K1 + 1, W - K2 + 1
        assert Ho_full > 0 and Wo_full > 0
        Ho, Wo = Ho_full // self.stride, Wo_full // self.stride
        assert Ho > 0 and Wo > 0
        out = array_api.array(
            numpy.zeros(shape=(N*Ho*Wo, Cout), dtype=A.dtype), 
            dtype = A.dtype,
            device=A.device
        )

        # compute out = sum A[:, i:i+H-k+1:stride, j:j+W-K+1:stride, :] @ B[i, j]
        for i in range(K1):
            for j in range(K2):
                # get matrix A[:, i:i+H-k+1:stride, j:j+W-K+1:stride, :]
                Asub = A[:, i:i+Ho_full:self.stride, j:j+Wo_full:self.stride, :].compact()
                Asub = Asub.reshape(new_shape=(N*Ho*Wo, Cin))
                Bsub = B[i, j, :, :].compact().reshape(new_shape=(Cin, Cout))
                out[:,:] = out + Asub @ Bsub

        # reshape out then return
        out = out.reshape(new_shape=(N, Ho, Wo, Cout))
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs[0], node.inputs[1]
        assert W.shape[0] == W.shape[1], 'only square kernel supported'

        if self.padding > 0:
            X = pad_zero(X, self.pad_width)

        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride-1)

            # crop out_grad to shape of Conv(X, W, stride=1)
            height, width, kersize = X.shape[1], X.shape[2], W.shape[0]
            Ho, Wo = height - kersize + 1, width - kersize + 1

            deltaH = out_grad.shape[1] - Ho
            deltaW = out_grad.shape[2] - Wo
            assert deltaH >= 0 and deltaH < self.stride and deltaW >= 0 and deltaW < self.stride
            if deltaH > 0 or deltaW > 0:
                out_grad = unpad_zero(out_grad, ((0, 0), (0, deltaH), (0, deltaW), (0, 0)))

            assert out_grad.shape == (X.shape[0], Ho, Wo, W.shape[-1])

        Xgrad, Wgrad = Conv.gradient_s1_p0(out_grad, X, W)

        if self.padding > 0:
            Xgrad = unpad_zero(Xgrad, self.pad_width)

        return Xgrad, Wgrad
        ### END YOUR SOLUTION
    
    @staticmethod
    def gradient_s1_p0(A: Tensor, X: Tensor, W: Tensor):
        ''' gradient of convolution when stride = 1 and padding = 0

        Parameters
        ----------
        A : NDArray, shape: [N, Ho, Wo, Cout]
            The adjoin tensor to the output

        X : NDArray, shape: [N, H, W, Cin]
            The input tensor
            
        W : NDArray, shape: [K, K, Cin, Cout]
            Convolution parameters

        Returns
        -------
        Tuple[NDArray, NDArray]
            The gradients with respect to the input tensor X and the convolution parameters W.
        '''
        # compute DW
        DW = conv(Permute((3, 1, 2, 0))(X), Permute((1,2,0,3))(A), padding=0)
        DW = Permute((1, 2, 0, 3))(DW)
        assert DW.shape == W.shape, f'DW.shape={DW.shape}, W.shape={W.shape}'
        
        # compute DX
        K = W.shape[0]
        DX = conv(
            A,
            Permute((0,1,3,2))(flip(W, (0, 1))),
            padding=K-1
        )
        assert DX.shape == X.shape

        return DX, DW

def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


