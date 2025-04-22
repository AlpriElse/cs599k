cmake_minimum_required(VERSION 3.10)

project(CSE599 LANGUAGES CXX CUDA)

# Language standards and compiler settings
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "80;90")

# Set optimizations and other flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG -Xcompiler=-fPIE -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(
  silu
  main.cu
  silu.cu
)

