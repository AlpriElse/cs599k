#include <cuda_runtime.h>
#include <stdio.h>
#include "silu.h"

// SiLU activation function: x * sigmoid(x)
__device__ __forceinline__ float silu_activation(float x) {
    return x / (1.0f + expf(-x));
}

// SiLU kernel that processes one row per block with tiled processing within each row
__global__ void silu_kernel(float *input, float *output, int matrix_size) {
    // Each block processes one row
    const int row = blockIdx.x;
    const int col_start = threadIdx.x;
    const int stride = blockDim.x;
    
    // Process elements in the row with strided access pattern
    for (int col = col_start; col < matrix_size; col += stride) {
        int idx = row * matrix_size + col;
        output[idx] = silu_activation(input[idx]);
    }
}

// Optimized 2D grid kernel for SiLU activation
// Each thread block processes a 2D tile of the matrix
__global__ void silu_kernel_2d(float *input, float *output, int matrix_size) {
    // Calculate 2D indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Calculate global indices
    const int col = bx * blockDim.x + tx;
    const int row = by * blockDim.y + ty;
    
    // Check if within bounds
    if (row < matrix_size && col < matrix_size) {
        const int idx = row * matrix_size + col;
        output[idx] = silu_activation(input[idx]);
    }
}

// Vector load version for 4-element processing per thread
__global__ void silu_kernel_vector(float *input, float *output, int matrix_size) {
    // Calculate base index
    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * blockDim.x * 4;
    const int idx = block_start + tid * 4;
    
    // Process 4 elements at once if possible
    if (idx + 3 < matrix_size * matrix_size) {
        float4 in_vec = *((float4*)&input[idx]);
        
        // Apply SiLU to each element
        float out1 = silu_activation(in_vec.x);
        float out2 = silu_activation(in_vec.y);
        float out3 = silu_activation(in_vec.z);
        float out4 = silu_activation(in_vec.w);
        
        // Store results
        output[idx] = out1;
        output[idx+1] = out2;
        output[idx+2] = out3;
        output[idx+3] = out4;
    }
    // Handle remaining elements
    else if (idx < matrix_size * matrix_size) {
        for (int i = 0; i < 4; i++) {
            if (idx + i < matrix_size * matrix_size) {
                output[idx + i] = silu_activation(input[idx + i]);
            }
        }
    }
}

void silu(float *input, float *output, int matrix_size, int block_size, int kernel_type) {
    // Set grid dimensions for original kernel
    dim3 grid(matrix_size, 1, 1);
    dim3 block(block_size, 1, 1);
    
    // Launch kernels based on the chosen type
    if (kernel_type == 1) {
        // 2D grid configuration
        dim3 block_2d(16, 16, 1);  // 16x16 threads per block
        dim3 grid_2d((matrix_size + block_2d.x - 1) / block_2d.x, 
                     (matrix_size + block_2d.y - 1) / block_2d.y, 1);
        
        silu_kernel_2d<<<grid_2d, block_2d>>>(input, output, matrix_size);
    }
    else if (kernel_type == 0) {
        // Original kernel
        silu_kernel<<<grid, block>>>(input, output, matrix_size);
    }
    else {
        // Vector processing configuration (default)
        int total_elements = matrix_size * matrix_size;
        int elements_per_thread = 4;
        int threads_per_block = block_size;
        int total_threads_needed = (total_elements + elements_per_thread - 1) / elements_per_thread;
        int blocks_needed = (total_threads_needed + threads_per_block - 1) / threads_per_block;
        
        silu_kernel_vector<<<blocks_needed, threads_per_block>>>(input, output, matrix_size);
    }
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Check for kernel execution errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
    }
}
