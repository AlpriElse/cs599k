#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <math.h>
#include <stdint.h>
#include "rms_norm_vector.h"

namespace cg = cooperative_groups;

// Implementation flag (0=auto, 1=original, 2=float4)
int rms_norm_implementation = 0;

// Original CUDA kernel to compute RMS normalization for vectors
__global__ void rms_norm_vector_kernel(float *input, float *weight, float *output, int cols, float epsilon) {
    int vec_idx = blockIdx.x;  // Each block handles one vector
    int tid = threadIdx.x;     // Thread ID within the block
    
    extern __shared__ float sdata[];
    
    // Each thread computes sum of squares for its portion of the vector
    float thread_sum_squared = 0.0f;
    
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = input[vec_idx * cols + i];
        thread_sum_squared += val * val;
    }
    
    // Store partial sum in shared memory
    sdata[tid] = thread_sum_squared;
    __syncthreads();
    
    // Parallel reduction to compute sum of squares
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute RMS normalization factor (only done by thread 0)
    float rms;
    if (tid == 0) {
        rms = sqrt(sdata[0] / (float)cols + epsilon);
        sdata[0] = rms;  // Store for other threads to use
    }
    __syncthreads();
    
    // All threads read the rms value
    rms = sdata[0];
    
    // Each thread normalizes its portion of the vector
    for (int i = tid; i < cols; i += blockDim.x) {
        int idx = vec_idx * cols + i;
        output[idx] = (input[idx] / rms) * weight[i];
    }
}

// Optimized kernel using float4 for vectorized memory access
__global__ void rms_norm_vector_kernel_float4(float *input, float *weight, float *output, int cols, float epsilon) {
    int vec_idx = blockIdx.x;  // Each block handles one vector
    int tid = threadIdx.x;     // Thread ID within the block
    
    extern __shared__ float sdata[];
    
    // Each thread computes sum of squares for its portion of the vector
    float thread_sum_squared = 0.0f;
    
    // Process elements in chunks of 4 using float4
    int cols_float4 = cols / 4;
    
    for (int i = tid; i < cols_float4; i += blockDim.x) {
        // Load 4 values at once using float4
        float4 input_vec = reinterpret_cast<float4*>(&input[vec_idx * cols])[i];
        
        // Compute sum of squares for each component
        thread_sum_squared += input_vec.x * input_vec.x;
        thread_sum_squared += input_vec.y * input_vec.y;
        thread_sum_squared += input_vec.z * input_vec.z;
        thread_sum_squared += input_vec.w * input_vec.w;
    }
    
    // Handle remaining elements (if cols is not a multiple of 4)
    int remaining_start = cols_float4 * 4;
    for (int i = remaining_start + tid; i < cols; i += blockDim.x) {
        float val = input[vec_idx * cols + i];
        thread_sum_squared += val * val;
    }
    
    // Store partial sum in shared memory
    sdata[tid] = thread_sum_squared;
    __syncthreads();
    
    // Parallel reduction to compute sum of squares
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute RMS normalization factor (only done by thread 0)
    float rms;
    if (tid == 0) {
        rms = sqrt(sdata[0] / (float)cols + epsilon);
        sdata[0] = rms;  // Store for other threads to use
    }
    __syncthreads();
    
    // All threads read the rms value
    rms = sdata[0];
    
    // Process vector elements in chunks of 4 for output
    for (int i = tid; i < cols_float4; i += blockDim.x) {
        int base_idx = i * 4;
        
        // Load 4 input values at once
        float4 input_vec = reinterpret_cast<float4*>(&input[vec_idx * cols])[i];
        
        // Load 4 weight values
        float w0 = weight[base_idx];
        float w1 = weight[base_idx + 1];
        float w2 = weight[base_idx + 2];
        float w3 = weight[base_idx + 3];
        
        // Compute 4 normalized and weighted values
        float4 output_vec;
        output_vec.x = (input_vec.x / rms) * w0;
        output_vec.y = (input_vec.y / rms) * w1;
        output_vec.z = (input_vec.z / rms) * w2;
        output_vec.w = (input_vec.w / rms) * w3;
        
        // Store 4 results at once
        reinterpret_cast<float4*>(&output[vec_idx * cols])[i] = output_vec;
    }
    
    // Handle remaining elements (if cols is not a multiple of 4)
    for (int i = remaining_start + tid; i < cols; i += blockDim.x) {
        int idx = vec_idx * cols + i;
        output[idx] = (input[idx] / rms) * weight[i];
    }
}

// Check if the pointer is aligned for float4 operations (16-byte alignment)
bool is_aligned_float4(const void* ptr) {
    return ((uintptr_t)ptr % 16) == 0;
}

void rms_norm_vector(float *input, float *weight, float *output, int cols, float epsilon) {
    // Determine number of vectors based on the input shape (1, batch_size, cols)
    const int batch_size = 1024;  // As specified in the problem
    
    // Device memory pointers
    float *d_input, *d_weight, *d_output;
    
    // Calculate total memory size
    size_t input_size = batch_size * cols * sizeof(float);
    size_t weight_size = cols * sizeof(float);
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_weight, weight_size);
    cudaMalloc((void**)&d_output, input_size);
    
    // Copy input data to device
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = batch_size;  // One block per vector
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    
    // Check float4 alignment and suitability
    bool float4_suitable = (cols % 4 == 0) && 
                           is_aligned_float4(d_input) && 
                           is_aligned_float4(d_weight) && 
                           is_aligned_float4(d_output);
    
    // Determine which implementation to use based on flag
    bool use_float4 = false;
    
    switch (rms_norm_implementation) {
        case 0: // Auto
            use_float4 = float4_suitable;
            break;
        case 1: // Original
            use_float4 = false;
            break;
        case 2: // Float4
            // Only use float4 if it's suitable
            use_float4 = float4_suitable;
            if (!float4_suitable) {
                printf("Warning: Float4 implementation requested but not suitable. Using original implementation.\n");
            }
            break;
        default:
            printf("Warning: Unknown implementation flag %d. Using auto selection.\n", rms_norm_implementation);
            use_float4 = float4_suitable;
    }
    
    // Launch the appropriate kernel
    if (use_float4) {
        rms_norm_vector_kernel_float4<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_input, d_weight, d_output, cols, epsilon);
    } else {
        rms_norm_vector_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_input, d_weight, d_output, cols, epsilon);
    }
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(output, d_output, input_size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}
