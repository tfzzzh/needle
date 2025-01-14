cmake_minimum_required(VERSION 3.5)
project(needle C CXX)
cmake_policy(SET CMP0146 OLD)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# find correct version of Python
execute_process(COMMAND python3-config --prefix
  OUTPUT_VARIABLE Python_ROOT_DIR)
find_package(Python COMPONENTS Development Interpreter REQUIRED)
message(STATUS "Python found at: ${Python_EXECUTABLE}, include dir: ${Python_INCLUDE_DIRS}")
include_directories(${Python_INCLUDE_DIRS})
message(STATUS Python_ROOT_DIR=${Python_ROOT_DIR})

# # find pybind
# message(STATUS "Python Executable: ${Python_EXECUTABLE}")
# # execute_process(COMMAND python3 -m pybind11 --cmakedir
# #   RESULT_VARIABLE __pybind_exit_code
# #   OUTPUT_VARIABLE __pybind_path
# #   OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "check python:")
# execute_process(COMMAND ${Python_EXECUTABLE} -m pip --version)
execute_process(COMMAND pwd)
execute_process(COMMAND ${Python_EXECUTABLE} -m pip show pybind11)
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Execution results: ${__pybind_exit_code}, pybind_path ${__pybind_path}")
find_package(pybind11 PATHS ${__pybind_path} REQUIRED)
message(__pybind_path=${__pybind_path})

# message(STATUS "find pybind package at ${PYBIND_CMAKE_DIR}")
# find_package(pybind11 PATH ${PYBIND_CMAKE_DIR} REQUIRED)


if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
add_library(ndarray_backend_cpu MODULE ndarray_backend_cpu.cc)
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
pybind11_extension(ndarray_backend_cpu)
pybind11_strip(ndarray_backend_cpu)


# directly output to ffi folder
# set_target_properties(ndarray_backend_cpu
#   PROPERTIES
#   LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/needle/backend_ndarray
#   CXX_VISIBILITY_PRESET "hidden"
# )

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()



# ####################
# ### CUDA BACKEND ###
# ####################
# find_package(CUDA)
# if(CUDA_FOUND)
#   message(STATUS "Found cuda, building cuda backend")

#   include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
#   list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

#   # invoke nvidia smi to detect if we really have a GPU
#   execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
#   if(NV_RET EQUAL "0")
#     CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
#   else()
#     # set to 3.7 the flag of K80
#     CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
#   endif()

#   # set arch flags properly
#   CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})

#   target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
#   pybind11_extension(ndarray_backend_cuda)
#   pybind11_strip(ndarray_backend_cuda)

# #   # directly output to ffi folder
# #   set_target_properties(ndarray_backend_cuda
# #     PROPERTIES
# #     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/needle/backend_ndarray
# #     CXX_VISIBILITY_PRESET "hidden"
# #     CUDA_VISIBILITY_PRESET "hidden"
# # )

# endif()

