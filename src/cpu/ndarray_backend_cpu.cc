#include <cstddef>
#include <cstdint>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size; // length of the array
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

template<typename T>
void print_vector(const std::vector<T> & vec) {
  for (const T & elem : vec)
    std::cout << elem << " ";
  std::cout << std::endl;
}
// prod(shape)
size_t array_size(const std::vector<int32_t> & shape) {
  assert (shape.size() > 0);
  size_t num_elem = 1;

  for (int32_t dim : shape) {
    num_elem *= dim;
  }

  return num_elem;
}

// increase index and return carry
int incc_index(const std::vector<int32_t> & shape, std::vector<int32_t> & index) {
  assert(shape.size() == index.size());
  int n = index.size();
  assert (n > 0);

  int carry = 1;
  // loop from last dimension to 0, once a dimension is full carry -> 1
  for (int dim = n - 1; dim >= 0; --dim) {
    index[dim] += carry;
    // std::cout << "index[dim]=" << index[dim] << std::endl;

    // perform plus and carrying when overflow
    if (index[dim] >= shape[dim]) {
      carry = 1;
      index[dim] = 0;
    }
    else {
      carry = 0;
    }
  }

  // for (int j=0; j < n; ++j) std::cout << index[j] << ",";
  // std::cout << std::endl;
  return carry;
}

// given index, strides, offset -> the linear index
size_t get_linear_index(const std::vector<int32_t> & index, const std::vector<int32_t> & strides, size_t offset) {
  size_t result = 0;
  for (size_t i = 0; i < index.size(); ++i) {
    result += ((size_t) index[i]) * strides[i];
  }
  return result + offset;
}

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  // assumption: shape is not empty
  assert (shape.size() > 0);

  // check if out array have size == prod(shape)
  size_t num_elems = array_size(shape);
  assert (out->size == num_elems);

  // for each index from [0, 0, ...] to [shape[0]-1, ...] fill the AlignedArray
  int ndim = shape.size();
  std::vector<int32_t> index(ndim, 0);

  for (size_t i = 0; i < num_elems; ++i) {
    //for (int j=0; j < ndim; ++j) std::cout << index[j] << ",";
    // std::cout << get_linear_index(index, strides, offset) << std::endl;
    out->ptr[i] = a.ptr[get_linear_index(index, strides, offset)];
    int carry = incc_index(shape, index);

    if (i < num_elems - 1) assert(carry == 0);
    if (i == num_elems - 1) assert(carry == 1);
  }

  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  assert (a.size == array_size(shape));

  // foreach item in a assign it to correspond index in out
  int ndim = shape.size();
  std::vector<int32_t> index(ndim, 0);
  // std::cout << "size: " << a.size << std::endl;
  // std::cout << "size of out:" << out->size << std::endl;
  // std::cout << "shape:" << std::endl;
  // print_vector(shape);


  for (size_t i = 0; i < a.size; ++i) {
    // std::cout << "set " << get_linear_index(index, strides, offset) << " to " << a.ptr[i] << std::endl;
    out->ptr[get_linear_index(index, strides, offset)] = a.ptr[i];

    // update index and test if it is overflow
    int carry = incc_index(shape, index);
    if (i < a.size - 1) assert(carry == 0);
    if (i == a.size - 1) assert(carry == 1);
  }
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  assert(size == array_size(shape));
  int ndim = shape.size();
  std::vector<int32_t> index(ndim, 0);
  for (size_t i = 0; i < size; ++i) {
    out->ptr[get_linear_index(index, strides, offset)] = val;

    // update index and test if it is overflow
    int c = incc_index(shape, index);
    if (i < size - 1) assert(c == 0);
    if (i == size - 1) assert(c == 1);
  }
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */
 
// pairwise operation
void EwisePairOp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, std::function<scalar_t(scalar_t, scalar_t)> op) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], b.ptr[i]);
  }
}

void ScalarPairOp(const AlignedArray& a, scalar_t val, AlignedArray* out, std::function<scalar_t(scalar_t, scalar_t)> op) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], val);
  }
}

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwisePairOp(a, b, out, [](scalar_t a, scalar_t b){return a * b;});
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarPairOp(a, val, out, [](scalar_t a, scalar_t b){return a * b;});
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwisePairOp(a, b, out, [](scalar_t a, scalar_t b){return a / b;});
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarPairOp(a, val, out, [](scalar_t a, scalar_t b){return a / b;});
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarPairOp(a, val, out, [](scalar_t a, scalar_t b){return pow(a, b);});
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwisePairOp(a, b, out, [](scalar_t a, scalar_t b){return std::max(a, b);});
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarPairOp(a, val, out, [](scalar_t a, scalar_t b){return std::max(a, b);});
}

void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwisePairOp(a, b, out, [](scalar_t a, scalar_t b){return scalar_t(a == b);});
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarPairOp(a, val, out, [](scalar_t a, scalar_t b){return scalar_t(a == b);});
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwisePairOp(a, b, out, [](scalar_t a, scalar_t b){return scalar_t(a >= b);});
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarPairOp(a, val, out, [](scalar_t a, scalar_t b){return scalar_t(a >= b);});
}

// unary operation
void EwiseOp(const AlignedArray& a, AlignedArray* out, std::function<scalar_t(scalar_t)> op) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i]);
  }
}

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  EwiseOp(a, out, [](scalar_t x){return log(x);});
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  EwiseOp(a, out, [](scalar_t x){return exp(x);});
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  EwiseOp(a, out, [](scalar_t x){return tanh(x);});
}


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  // check size of matrix
  assert(a.size == size_t(m) * n);
  assert(b.size == size_t(n) * p);
  assert(out->size == size_t(m) * p);

  // initialize out to 0
  for (size_t i=0; i < out->size; ++i)
    out->ptr[i] = 0.0;

  // matrix multiplier using order k, i, j
  for (size_t k=0; k < n; ++k) {
    for (size_t i=0; i < m; ++i) {
      size_t ip = i * p;
      size_t kp = k * p;
      size_t in = i * n;
      for (size_t j=0; j < p; ++j) {
        out->ptr[ip + j] += a.ptr[in + k] * b.ptr[kp + j];
      }
    }
  }

  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (int i=0; i < TILE; ++i) {
    int iTile = i * TILE;
    for (int k=0; k < TILE; ++k) {
      int kTile = k * TILE;
      for (int j=0; j < TILE; ++j) {
        out[iTile + j] += a[iTile + k] * b[kTile + j];
      }
    }
  }
  /// END SOLUTION
}

inline void zero_tile(scalar_t* out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */
  // out = (scalar_t*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (int i=0; i < TILE; ++i) {
    int iTile = i * TILE;
      for (int j=0; j < TILE; ++j) {
        out[iTile + j] = 0.0;
      }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  assert (m % TILE == 0 && n % TILE == 0 && p % TILE == 0 && "rows and cols must be a multiple of tile");
  uint32_t m_out = m / TILE, n_out = n / TILE, p_out = p / TILE;

  // O[i][j] = sum_k A[i][k] @ B[k][j]
  for (size_t i=0; i < m_out; ++i) {
    for (size_t j=0; j < p_out; ++j) {
      // set out[i][j] = 0
      scalar_t * out_tile = out->ptr + (i * p_out + j) * TILE * TILE;
      zero_tile(out_tile);
      for (size_t k=0; k < n_out; ++k) {
        scalar_t * a_tile = a.ptr + (i * n_out + k) * TILE * TILE;
        scalar_t * b_tile = b.ptr + (k * p_out + j) * TILE * TILE;
        AlignedDot(a_tile, b_tile, out_tile);
      }
    }
  } 
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  assert (a.size == out->size * reduce_size);
  assert (reduce_size > 0);
  for (size_t i=0; i < out->size; ++i) {
    size_t start = i * reduce_size;
    size_t end = (i+1) * reduce_size;

    scalar_t reduced = std::numeric_limits<scalar_t>::lowest();
    for (int j=start; j < end; ++j) {
      reduced = std::max(reduced, a.ptr[j]);
    }

    out->ptr[i] = reduced;
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  assert (a.size == out->size * reduce_size);
  assert (reduce_size > 0);
  for (size_t i=0; i < out->size; ++i) {
    size_t start = i * reduce_size;
    size_t end = (i+1) * reduce_size;

    scalar_t reduced = 0.0;
    for (int j=start; j < end; ++j) {
      reduced  += a.ptr[j];
    }

    out->ptr[i] = reduced;
  }
  /// END SOLUTION
}

// hw4
/*
def flip(a, out, shape, axis):
    out.array[:] = np.flip(a.array[:].reshape(shape), axis).reshape(-1)
*/
std::vector<int32_t> shape2stride(const std::vector<int32_t>& shape) {
  std::vector<int32_t> strides(shape.size(), 1);
  for (int i=shape.size()-1; i >= 1; --i) {
    strides[i-1] = shape[i] * strides[i];
  }
  return strides;
}

void Flip(const AlignedArray & a, AlignedArray * out,  std::vector<int32_t> shape, std::vector<int32_t> axis) {
  assert (a.size == out->size);
  assert (a.size == array_size(shape));

  // compute shape to stride
  std::vector<int32_t> strides = shape2stride(shape);

  std::vector<int32_t> index(shape.size(), 0);
  for (size_t i = 0; i < out->size; ++i) {
    std::vector<int32_t> index_flip = index;
    for (auto dim : axis) {
      index_flip[dim] = shape[dim] - index[dim] - 1;
      assert (index_flip[dim] >= 0);
    }

    out->ptr[i] = a.ptr[get_linear_index(index_flip, strides, 0)];
    incc_index(shape, index);
  }
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
  m.def("flip", Flip);
}
