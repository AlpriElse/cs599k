#include <cuda_runtime.h>
#include "silu.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <torch/extension.h>

namespace py = pybind11;

void py_silu(py::array_t<float> input, py::array_t<float> output) {
    py::buffer_info input_buf = input.request();
    py::buffer_info output_buf = output.request();

    float *input_data = static_cast<float*>(input_buf.ptr);
    float *output_data = static_cast<float*>(output_buf.ptr);

    int n = input_buf.shape[0];
    silu(input_data, output_data, n);
}

PYBIND11_MODULE(silu, m) {
    m.def("py_silu", &py_silu, "A function that adds two numbers");
}

