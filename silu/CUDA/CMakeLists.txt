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


# Pybind11
set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Using Python ${Python_EXECUTABLE} ${Python_VERSION} ")
add_subdirectory(
    extern/pybind11
)

pybind11_add_module(silu bindings.cpp silu.cu)

install(TARGETS silu DESTINATION .)