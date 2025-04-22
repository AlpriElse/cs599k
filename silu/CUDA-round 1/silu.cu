#include <cuda_runtime.h>
#include <stdio.h>

__global__ void silu_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] / (1 + exp(-input[idx]));
    }

}

void silu(float *input, float *output, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    silu_kernel<<<numBlocks, blockSize>>>(input, output, n);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed with error \"%s\"\n", cudaGetErrorString(err));
    }
}
