import operator
import math
import builtins
from functools import reduce
import numpy as np
from typing import List
from . import ndarray_backend_numpy
from . import ndarray_backend_cpu


# math.prod not in Python 3.7
def prod(x):
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device, wrapps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda():
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy():
    """Return numpy device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device():
    return cpu_numpy()


def all_devices():
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy()]


class NDArray:
    """A generic ND array class that may contain multipe different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.

    This class will only contains those functions that you need to implement
    to actually get the desired functionality for the programming examples
    in the homework, and no more.

    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """

    def __init__(self, other, device=None):
        """Create by copying another NDArray, or from numpy"""
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other):
        self._shape = other._shape      # tuple
        self._strides = other._strides  # tuple
        self._offset = other._offset    # integer
        self._device = other._device    # device
        self._handle = other._handle    # array pointer

    @staticmethod
    def compact_strides(shape):
        """Utility function to compute compact strides"""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """Create a new NDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    ### Properies and string representations
    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only support float32 for now
        return "float32"

    @property
    def ndim(self):
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)

    def __repr__(self):
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self):
        return self.numpy().__str__()

    ### Basic array manipulation
    def fill(self, value):
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device):
        """Convert between devices, using to/from numpy calls as the unifying bridge."""
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self):
        """convert to a numpy array"""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self):
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )

    def compact(self):
        """Convert a matrix to be compact"""
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape, strides):
        """Restride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle, offset=self._offset
        )

    @property
    def flat(self):
        return self.reshape((self.size,))

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.

        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.

        Args:
            new_shape (tuple): new shape of the array

        Returns:
            NDArray : reshaped array; this will point to thep
        """

        ### BEGIN YOUR SOLUTION
        if prod(new_shape) != prod(self.shape):
            raise ValueError(f"new_shape {new_shape} does not have the same element with old shape {self.shape}") 

        if not self.is_compact():
            raise ValueError("Array is not compact")

        _shape = tuple(new_shape)
        _strides = self.compact_strides(_shape)

        view = self.make(_shape, _strides, self.device, self._handle, self._offset)
        return view
        ### END YOUR SOLUTION

    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permuation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memroy as the original array.

        Args:
            new_axes (tuple): permuation order of the dimensions

        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """

        ### BEGIN YOUR SOLUTION
        # check if new_axes is all unique
        assert len(new_axes) == len(set(new_axes)), 'permute axes shall be unique'

        _shape = tuple(self._shape[i] for i in new_axes)
        _strides = tuple(self._strides[i] for i in new_axes)

        view = self.make(_shape, _strides, self.device, self._handle, self._offset)
        return view
        ### END YOUR SOLUTION

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.

        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1

        Args:
            new_shape (tuple): shape to broadcast to

        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """

        ### BEGIN YOUR SOLUTION
        # new_shape should longer or equal to the length of old shape
        dim_new, dim_old = len(new_shape), len(self._shape)
        assert dim_new >= dim_old

        # old_shape should convertable to new_shape
        shape_bdc = []
        stride_bdc = []
        
        # iterate over each dimension of shape_new in reverse order
        for i in range(dim_new-1, -1, -1):
            j = dim_old + (i-dim_new)

            # need broad cast
            if j < 0 or new_shape[i] != self._shape[j]:
                assert j < 0 or self._shape[j] == 1
                stride_bdc.append(0)
            
            # not need
            else:
                stride_bdc.append(self._strides[j])
            
            shape_bdc.append(new_shape[i])

        # reverse shape_bdc, stride_bdc
        shape_bdc.reverse()
        stride_bdc.reverse()

        # transform shape and stride
        # self._shape = tuple(shape_bdc)
        # self._strides = tuple(stride_bdc)

        view = self.make(tuple(shape_bdc), tuple(stride_bdc), self.device, self._handle, self._offset)
        return view
        ### END YOUR SOLUTION

    ### Get and set elements

    def process_slice(self, sl, dim):
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"

        # 0 <= start < stop <= self.shape[dim]
        assert start >= 0 and stop <= self.shape[dim]
        return slice(start, stop, step)

    def __getitem__(self, idxs):
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.

        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory

        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.

        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            coresponding to the subset of the matrix to get

        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memroy but just
            manipulate the shape/strides/offset of the new array, referecing
            the same array as the original one.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        ### BEGIN YOUR SOLUTION
        # for a slice (start, stop, step) the implicit shape is: ceil((stop - start) / step)
        new_shape = tuple(
            (s.stop - s.start + s.step - 1) // s.step for s in idxs
        )

        # let B = A[idxs]
        # B[i1, i2] = A[row[i1], col[i2]]
        # index[row[i1], col[i2]] = offset + row[i1] * stride[i1] + col[i2] * stride[i2]
        #   = offset + (start_1 + i1 * step1) * stride[0] + (start_2 + i2 * step2) * stride[1]
        #   = (offset + start_1 * stride[0] + start2 * stride[1]) + i1 * (step1 *stride[0]) + i2 * (step2 * stride[1])
        _offset = self._offset + builtins.sum(s.start * stride for (s, stride) in zip(idxs, self._strides))
        _strides = tuple(s.step * stride for (s, stride) in zip(idxs, self._strides))

        # init a view
        view = self.make(new_shape, _strides, self.device, self._handle, _offset)

        return view
        ### END YOUR SOLUTION

    def __setitem__(self, idxs, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            # assert self.shape == other.shape, "operation needs two equal-sized arrays"
            if self.shape == other.shape:
                ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
            
            elif self.ndim == 0:
                view = NDArray.make(shape=(1,), device=self.device)
                view.fill(self.numpy().item())
                view = view.broadcast_to(other.shape)
                out = NDArray.make(view.shape, device=self.device)
                ewise_func(view.compact()._handle, other.compact()._handle, out._handle)
            
            elif other.ndim == 0:
                view = NDArray.make(shape=(1,), device=self.device)
                view.fill(other.numpy().item())
                view = view.broadcast_to(self.shape)
                ewise_func(self.compact()._handle, view.compact()._handle, out._handle)
            
            else:
                raise NotImplementedError

        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    ### Elementwise functions

    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other):
        """Matrix multplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.

        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will restride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size

        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """
        # special case: use numpy device
        if self.device.name == 'cpu_numpy':
            arr_left = self.compact()._handle.array.reshape(*self.shape)
            arr_right = other.compact()._handle.array.reshape(*other.shape)
            out = arr_left.astype('float64') @ arr_right
            return NDArray(out, self.device)


        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions, i.e., sum/max over all element or over given axis
    # def reduce_view_out(self, axis, keepdims=False):
    #     """ Return a view to the array set up for reduction functions and output array. """
    #     if isinstance(axis, tuple) and not axis:
    #         raise ValueError("Empty axis in reduce")

    #     if axis is None:
    #         view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
    #         #out = NDArray.make((1,) * self.ndim, device=self.device)
    #         out = NDArray.make((1,), device=self.device)

    #     else:
    #         if isinstance(axis, (tuple, list)):
    #             assert len(axis) == 1, "Only support reduction over a single axis"
    #             axis = axis[0]

    #         view = self.permute(
    #             tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
    #         )
    #         out = NDArray.make(
    #             tuple([1 if i == axis else s for i, s in enumerate(self.shape)])
    #             if keepdims else
    #             tuple([s for i, s in enumerate(self.shape) if i != axis]),
    #             device=self.device,
    #         )
    #     return view, out
    

    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, axis, keepdims=False):
        """ Return a view to the array set up for reduction functions and output array. """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            out = NDArray.make((1,), device=self.device)

        else:
            if not isinstance(axis, (tuple, list)):
                axis = (axis, )

            out_perm = tuple(a for a in range(self.ndim) if a not in axis)
            out_shape = tuple(self.shape[a] for a in range(self.ndim) if a not in axis)

            view = self.permute(
                out_perm + axis
            )

            num_col = prod(self.shape[a] for a in axis)
            view = view.compact()
            view = view.reshape(out_shape + (num_col,))

            if keepdims:
                out_shape = list(self.shape)
                for dim in axis:
                    out_shape[dim] = 1
                out_shape = tuple(out_shape)

            out = NDArray.make(
                out_shape,
                device=self.device,
            )

        return view, out

    def sum(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def flip(self, axis):
        """
        Flip this ndarray along the specified axes.

        when axis = 0, aflip[i, j] = a[n-i, j]
        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which to flip over. The default,
            axis=None, will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.

            If axis is a tuple of ints, flipping is performed on all of the axes
            specified in the tuple.

        Returns
        -------
        out : a new array

        Note: compact() before returning.
        """
        ### BEGIN YOUR SOLUTION
        # assumption dimension of the array > 0
        shape = self.shape
        ndim = len(shape)
        assert ndim > 0, 'array with shape () is not supported'

        if axis is None:
            axis = tuple(range(ndim))

        # make axis to tuple
        if not isinstance(axis, tuple):
            axis = (axis, )
        
        # make axis increasing
        axis = tuple(sorted(axis))

        # check axis with bound
        assert axis[0] >= 0 and axis[-1] < ndim

        ### END YOUR SOLUTION
        # allocate out
        out = self.make(shape, device=self.device)

        # call kernel
        self.device.flip(self.compact()._handle, out._handle, shape, axis)

        return out

    def pad(self, axes):
        """
        Pad this ndarray by zeros by the specified amount in `axes`,
        which lists for _all_ axes the left and right padding amount, e.g.,
        axes = ( (0, 0), (1, 1), (0, 0)) pads the middle axis with a 0 on the left and right side.

        Parameters
        ----------
        axes : list of tuples
            Each tuple contains two integers representing the padding before and after the axis.

        Returns
        -------
        NDArray
            A new NDArray with the specified padding.
        """
        ### BEGIN YOUR SOLUTION
        assert self.ndim == len(axes)

        # expand shape using axes
        shape_new = list(self.shape)
        for dim in range(self.ndim):
            shape_new[dim] += (axes[dim][0] + axes[dim][1])

        # allocate an array and then fill it with 0
        arr_new = self.make(shape_new, device=self.device)
        arr_new.fill(0.0)

        # set value of the newed array
        indices = []
        for i in range(self.ndim):
            indices.append(slice(axes[i][0], shape_new[i] - axes[i][1]))

        arr_new[tuple(indices)] = self

        return arr_new
        ### END YOUR SOLUTION

def array(a, dtype="float32", device=None):
    """Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)


def reshape(array, new_shape):
    return array.reshape(new_shape)


def maximum(a, b):
    return a.maximum(b)


def log(a):
    return a.log()


def exp(a):
    return a.exp()


def tanh(a):
    return a.tanh()


def sum(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)

def max(a, axis=None, keepdims=False):
    return a.max(axis=axis, keepdims=keepdims)

def flip(a, axes):
    return a.flip(axes)

def transpose(a, axies=None):
    if axies is None:
        axies = list(range(a.shape))
        axies = tuple(axies[::-1])

    return a.permute(axies)

def power(a: NDArray, b: NDArray):
    if not isinstance(b, NDArray):
        return a ** float(b)

    elif (len(b.shape) == 0):
        return a ** float(b.numpy())
    
    else:
        raise NotImplementedError
    

def stack(arrs: List[NDArray], axis=0):
    """
    Join a sequence of arrays along a new axis.

    The ``axis`` parameter specifies the index of the new axis in the
    dimensions of the result. For example, if ``axis=0`` it will be the first
    dimension and if ``axis=-1`` it will be the last dimension.

    Parameters
    ----------
    arrays : sequence of NDArray
        Each array must have the same shape.

    axis : int, optional
        The axis in the result array along which the input arrays are stacked.

    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays.

    Examples
    --------
    >>> a = NDArray(np.array([1, 2, 3]))
    >>> b = NDArray(np.array([4, 5, 6]))
    >>> stack([a, b], axis=0)
    NDArray([[1, 2, 3],
             [4, 5, 6]])
    """
    # check if all element in arrs are ndarray
    assert len(arrs) > 0, "inputs shall not empty"
    for arr in arrs:
        assert isinstance(arr, NDArray), "all element in arrs must be NDArray"

    # check if all alements have the same size
    n = len(arrs)
    for i in range(1, n):
        assert arrs[i].shape == arrs[0].shape, 'shapes of all arrays must be the same'
        assert arrs[i].device == arrs[0].device

    # check axis <= dim of matrix
    ndim = len(arrs[0].shape)
    assert axis <= ndim

    # when axis == ndim we need to and new dim for arrays
    if axis == ndim:
        shape_bdr = list(arrs[0].shape)
        shape_bdr.append(1)
        shape_bdr = tuple(shape_bdr)
        arrs_reshape = []
        for i in range(n):
            arr_reshape = reshape(arrs[i].compact(), shape_bdr)
            arrs_reshape.append(arr_reshape)
        arrs = arrs_reshape

    # stack the arrays
    # compute shape of the out array
    shape_in = arrs[0].shape
    shape_out = list(shape_in)
    shape_out[axis] = shape_in[axis] * n
    shape_out = tuple(shape_out)

    # allocate memory for out array
    out = NDArray.make(shape_out, device=arrs[0].device)

    # assign each arr to out
    indices = [slice(0, sz) for sz in shape_out]
    for i in range(n):
        arr = arrs[i]
        start = i * shape_in[axis]
        end = (i+1) * shape_in[axis]
        indices[axis] = slice(start, end)

        out[tuple(indices)] = arr
    
    return out


def split(arr: NDArray, num_chunk: int, axis: int=0):
    """
    Split an array into multiple equal_size sub-arrays as views into `arr`.

    Parameters
    ----------
    arr : NDArray
        Array to be divided into sub-arrays.
    num_chunk : int 
        the array will be divided
        into `num_chunk` equal arrays along `axis`.  If such a split is not possible,
        an error is raised.

    axis : int, optional
        The axis along which to split, default is 0.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of sub-arrays as views into `arr`.

    """
    shape = arr.shape

    # check num_chunk dividable by 
    size = shape[axis]
    if size % num_chunk != 0:
        raise ValueError("num_chunk must be divisible by axis size")
    
    # partition arr in axis
    submat = []
    indices = [slice(0, sz) for sz in shape]
    chunksize = size // num_chunk
    for i in range(num_chunk):
        indices[axis] = slice(i*chunksize, (i+1)*chunksize)
        submat.append(arr[tuple(indices)])
    
    return submat


def pad(arr, pad_width):
    return arr.pad(pad_width)


def sqrt(arr):
    return power(arr, 0.5)