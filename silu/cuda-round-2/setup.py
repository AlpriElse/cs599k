from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os
import torch
import glob


setup(
    name='cuda_extension',
    packages=find_packages(),
    install_requires=['torch'],
    ext_modules=[
        CppExtension(
            name='silu_cuda',
            sources=['silu_cuda/bindings.cpp'],
            extra_compile_args={"cxx": [
            "-O3",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
            ],
            'nvcc': ['-O2']},
            extra_link_args=[]
        ),
        CUDAExtension(
            name='cuda_extension',
            sources=['silu_cuda/silu.cu'],
            extra_compile_args={"cxx": [
                "-O3",
                "-fdiagnostics-color=always",
                "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
            ],
                                'nvcc': ['-O2']},
            extra_link_args=[]  
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)