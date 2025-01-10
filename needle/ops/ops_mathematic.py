"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api


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
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        '''
        d(A^B) = d(exp(log(A^B))) 
        '''
        A, B = node.inputs[0], node.inputs[1]
        GA = multiply(divide(node, A), B)
        GB = multiply(node, log(A))

        return multiply(out_grad, GA), multiply(out_grad, GB) 

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        scalar = array_api.array(self.scalar, dtype=a.dtype)
        return array_api.power(a, scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        '''
        out_grad * (scaler * x^{scaler - 1})
        '''
        ### BEGIN YOUR SOLUTION
        scalar = array_api.array(self.scalar, dtype=out_grad.dtype)
        return out_grad * mul_scalar(power_scalar(node.inputs[0], scalar-1), scalar)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)

def sqrt(a: Tensor):
    scalar = array_api.array(0.5, dtype=a.dtype)
    return power_scalar(a, scalar=scalar)

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
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide_scalar(out_grad, self.scalar)
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

        perm = array_api.arange(len(a.shape))
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


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
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
        grad = summation(out_grad, axes=self.axes)
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
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
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
    n = array_api.array(n, dtype=a.dtype)
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
        y = a.copy()
        y[y <= 0] = 0
        return y
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        '''
        d RelU(X) = 
            1. X > 0 -> 1
            2. X <= 0 -> 0
        '''
        ### BEGIN YOUR SOLUTION
        xarr = node.inputs[0].realize_cached_data()
        yarr = array_api.ones_like(xarr)
        yarr[xarr <= 0] = 0
        return multiply(out_grad, Tensor(yarr, requires_grad=False))
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


