cmake_minimum_required(VERSION 3.15...3.29)
project(example LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Using Python ${Python_EXECUTABLE} ${Python_VERSION} ")
add_subdirectory(
    extern/pybind11
)

pybind11_add_module(example example.cpp)
install(TARGETS example DESTINATION .)