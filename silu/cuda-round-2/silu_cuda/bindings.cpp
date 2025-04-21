#include "silu.h"

#include <torch/extension.h>
#include <stdio.h>

#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace silu_cuda {

void py_silu(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on CUDA device");
    
    float *input_data = input.data_ptr<float>();
    float *output_data = output.data_ptr<float>();
    
    int n = input.numel();
    silu(input_data, output_data, n);
}

TORCH_LIBRARY(silu_cuda, m) {
  m.def("silu(Tensor a, Tensor b) -> Tensor");
}
}