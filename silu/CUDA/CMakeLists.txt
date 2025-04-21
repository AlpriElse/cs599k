cmake_minimum_required(VERSION 3.15...3.29)

project(test LANGUAGES CXX CUDA)

# Language standards and compiler settings
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90)

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "90")
 set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[compute_90a,sm_90a] -Xcompiler=-fPIE -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

execute_process(
  COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
message(STATUS "Found PyTorch: ${TORCH_INCLUDE_DIRS}")
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# Pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Using Python ${Python_EXECUTABLE} ${Python_VERSION} ")
add_subdirectory(
    extern/pybind11
)

pybind11_add_module(silu bindings.cpp silu.cu)
target_include_directories(silu PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(silu PRIVATE ${TORCH_LIBRARIES})

install(TARGETS silu DESTINATION .)