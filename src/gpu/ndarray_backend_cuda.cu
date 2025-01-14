#include <cstddef>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
__device__ CudaVec compact_index_to_sub(size_t index, const CudaVec & shape) {
  CudaVec subs;
  subs.size = shape.size;
  for (int i = subs.size - 1; i >= 0; --i) {
    subs.data[i] = index % shape.data[i];
    index /= shape.data[i];
  }
  return subs;
}

__device__ size_t get_total_num_elements(const CudaVec & shape) {
  size_t n = 1;
  for (int i=0; i < shape.size; ++i) {
    n *= shape.data[i];
  }
  return n;
}

__device__ size_t sub_to_index(const CudaVec & subs, const CudaVec & strides, size_t offset) {
  size_t idx = offset;
  for (size_t i=0; i < subs.size; ++i) {
    idx += subs.data[i] * strides.data[i];
  }
  return idx;
}

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid >= size) return;

  // write out[gid]
  // using gid get subs from shape
  // CudaVec subs;
  // subs.size = shape.size;
  // size_t idx_linear = gid;
  // for (int i = subs.size - 1; i >= 0; --i) {
  //   subs.data[i] = idx_linear % shape.data[i];
  //   idx_linear /= shape.data[i];
  // }
  auto subs = compact_index_to_sub(gid, shape);

  // compute index of a using stride and subs
  // size_t idx_a = offset;
  // for (size_t i=0; i < subs.size; ++i) {
  //   idx_a += subs.data[i] * strides.data[i];
  // }
  size_t idx_a = sub_to_index(subs, strides, offset);

  // assign to out[gid]
  out[gid] = a[idx_a];

  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t n, CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid >= n) return;

  // get subs from gid
  auto subs = compact_index_to_sub(gid, shape);

  // use subs to find index of out
  auto idx_a = sub_to_index(subs, strides, offset);

  // set out
  out[idx_a] = a[gid];
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END SOLUTION
}


__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out, CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid >= size) return;

  // get subs from gid
  auto subs = compact_index_to_sub(gid, shape);

  // use subs to find index of out
  auto idx_a = sub_to_index(subs, strides, offset);

  // set out
  out[idx_a] = val;
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape), VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


// __global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
//   // Calculate the global index of the thread.
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid < size) out[gid] = a[gid] + b[gid];
// }
typedef scalar_t (*UnitaryOp)(scalar_t);
typedef scalar_t (*BinaryOp)(scalar_t, scalar_t);

// __device__ scalar_t log_scalar(scalar_t x) {return logf(x);}
// __device__ scalar_t exp_scalar(scalar_t x) {return expf(x);}
// __device__ scalar_t tanh_scalar(scalar_t x) {return (expf(x) - expf(-x)) / (expf(x) + expf(-x));}

__device__  scalar_t add_scalar(scalar_t a, scalar_t b) {return a + b;}
__device__  scalar_t mult_scalar(scalar_t a, scalar_t b) {return a * b;}
__device__  scalar_t div_scalar(scalar_t a, scalar_t b) {return a / b;}
__device__  scalar_t maximum_scalar(scalar_t a, scalar_t b) {return a >= b ? a : b;}
__device__  scalar_t eq_scalar(scalar_t a, scalar_t b) {return scalar_t(a == b);}
__device__  scalar_t ge_scalar(scalar_t a, scalar_t b) {return scalar_t(a >= b);}
__device__  scalar_t pow_scalar(scalar_t a, scalar_t b) {return powf(a, b);}


template<UnitaryOp op>
__global__ void EwiseUnitaryOpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid]);
}


template<BinaryOp op>
__global__ void EwisePairOpKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid], b[gid]);
}

template<UnitaryOp op>
void EwiseUnitaryOp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseUnitaryOpKernel<op><<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

template<BinaryOp op>
void EwiseBinaryOp(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwisePairOpKernel<op><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseBinaryOp<add_scalar>(a, b, out);
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseBinaryOp<mult_scalar>(a, b, out);
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseBinaryOp<div_scalar>(a, b, out);
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseBinaryOp<maximum_scalar>(a, b, out);
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseBinaryOp<eq_scalar>(a, b, out);
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseBinaryOp<ge_scalar>(a, b, out);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  EwiseUnitaryOp<logf>(a, out);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  EwiseUnitaryOp<expf>(a, out);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  EwiseUnitaryOp<tanhf>(a, out);
}

// __global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
//   // Calculate the global index of the thread.
//   // size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   // if (gid < size) out[gid] = a[gid] + b[gid];
//   EwisePairOpKernel(a, b, out, size, add_scalar);
// }


// void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
//   /**
//    * Add together two CUDA arrays.
//    * Args:
//    *   a: Input array 'a' to be added
//    *   b: Input array 'b' to be added
//    *   out: Output array to store the result of 'a + b'
//    */
//   CudaDims dim = CudaOneDim(out->size);

//   // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
//   EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
// }

template<BinaryOp op>
__global__ void ScalarPairOpKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid], val);
}

template<BinaryOp op>
void ScalarBinaryOp(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarPairOpKernel<op><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarBinaryOp<add_scalar>(a, val, out);
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarBinaryOp<mult_scalar>(a, val, out);
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarBinaryOp<div_scalar>(a, val, out);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarBinaryOp<pow_scalar>(a, val, out);
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarBinaryOp<maximum_scalar>(a, val, out);
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarBinaryOp<eq_scalar>(a, val, out);
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarBinaryOp<ge_scalar>(a, val, out);
}

// __global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
//   // Calculate the global index of the thread.
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid < size) out[gid] = a[gid] + val;
// }

// void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
//   /**
//    * Add a scalar value to every element of a CUDA array.
//    * Args:
//    *   a: Input array 'a'
//    *   val: Scalar value to be added
//    *   out: Output array to store the result of 'a + val'
//    */
//   CudaDims dim = CudaOneDim(out->size);

//   // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
//   // and store the result in array 'out'.
//   ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
// }

/**
 * In the code the follows, use the above template to create analogous elementise
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


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////
/*
For matrix multiplication C = A * B, partition C, A, B to submatrix of size BLOCK * BLOCK
Let C[i, j] index block && C(i, j) index elements

  C[i][j] = sum_k A[i][k] * B[k][j]

  C[i][j] is of size Block * Block located at [i*BLOCK : (i+1)*Block ]..
    index of C[ib][jb](i, j) -> C(ib*BLOCK + i, jb*BLOCK + j)
    linear index (ib*BLOCK + i) * cols + (jb*BLOCK + j)


corner case: when BLOCK is not dividable => ib * BLOCK + i may out of bound

thread partition:
  outter [m/block, n/block], inner [block * block]

  outer index: (ib, jb), inner index (i, j) for C

parameters:
  A matrix of size [m, p]
  B matrix of size [p, n]
  C matrix of size [m, n]
*/
const int BLOCK = 16;
__global__ void matmul_kernel(const scalar_t * A, const scalar_t * B, scalar_t * C, uint32_t m, uint32_t n, uint32_t p) {
  size_t ib = blockIdx.x, jb = blockIdx.y;
  size_t i = threadIdx.x, j = threadIdx.y;

  // compute C[ib][jb] using the threadblock of size BLOCK * BLOCK
  uint32_t pb = (p + BLOCK - 1) / BLOCK;
  scalar_t c = 0.0;

  // foreach A[ib][k] && B[k][jb] with k < pb, compute their matmul and sum to C
  for (size_t kb = 0; kb < pb; ++kb) {
    // load A[ib][kb] to mem using threads
    //  thread(i, j) loads A[ib][kb](i, j) to As[i][j]
    //  (i, j) is only valid when rowidx and colidx not out of bounded
    __shared__ scalar_t As[BLOCK][BLOCK];
    if (ib*BLOCK + i < m && kb*BLOCK + j < p)
      As[i][j] = A[(ib*BLOCK + i)*p + (kb*BLOCK + j)];
    else
      As[i][j] = 0.0;

    // load B[kb][jb] to mem using threads
    //  thread(i, j) loads B[kb][jb](i, j) to Bs[i][j]
    __shared__ scalar_t Bs[BLOCK][BLOCK];
    if (kb*BLOCK + i < p && jb*BLOCK + j < n)
      Bs[i][j] = B[(kb*BLOCK + i)*n + (jb*BLOCK + j)];
    else
     Bs[i][j] = 0.0;

    // wait for all thread done
    __syncthreads();

    // thread (i, j) compute sum A[ib][kb](i, k) * B[kb][jb](k, j)
    for (size_t k=0; k < BLOCK; ++k) {
      c += As[i][k] * Bs[k][j];
    }

    // make sure As, Bs is still avaiable for slow threads
    __syncthreads();
  }

  // write back this thread only manage C[ib][jb](i, j)
  if (ib*BLOCK + i < m && jb * BLOCK + j < n)
    C[(ib*BLOCK + i) * n + jb * BLOCK + j] = c;
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  // partition out of size m * p into task with grid_size (m / b, p / b) thread_size (b, b)
  dim3 grid((M + BLOCK - 1) / BLOCK, (P + BLOCK - 1) / BLOCK);
  dim3 block(BLOCK, BLOCK);

  matmul_kernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, P, N);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
__global__ void reduce_max_kernel(const scalar_t * a, scalar_t* out, size_t reduce_size, size_t n) {
  size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid >= n) return;

  // out[gid] = reduce_max(a[gid*reduce_size : (gid+1) * reduce_size])
  scalar_t result = -FLT_MAX;
  for (size_t i = gid * reduce_size; i < (gid+1) * reduce_size; ++i) {
    result = max(result, a[i]);
  }

  out[gid] = result;
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  reduce_max_kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}


__global__ void reduce_sum_kernel(const scalar_t * a, scalar_t* out, size_t reduce_size, size_t n) {
  size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid >= n) return;

  // out[gid] = reduce_max(a[gid*reduce_size : (gid+1) * reduce_size])
  scalar_t result = 0.0;
  for (size_t i = gid * reduce_size; i < (gid+1) * reduce_size; ++i) {
    result += a[i];
  }

  out[gid] = result;
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  reduce_sum_kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
