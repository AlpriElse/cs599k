#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "rms_norm_matrix.h"

// Implementation flag (0=auto, 1=original, 2=float4)
int rms_norm_implementation = 0;

// Original CUDA kernel to compute RMS normalization for each row of the matrix
// Uses shared memory and warp-level operations
__global__ void rms_norm_matrix_kernel(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    extern __shared__ float sdata[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each block processes one row, with multiple threads per row
    if (row < rows) {
        float thread_sum_squared = 0.0f;
        
        // Each thread computes sum of squares for its portion of the row
        for (int col = tid; col < cols; col += blockDim.x) {
            float val = input[row * cols + col];
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
            sdata[0] = rms; // Store rms in shared memory for other threads to use
        }
        __syncthreads();
        
        // All threads read the rms value
        rms = sdata[0];
        
        // Each thread normalizes its portion of the row
        for (int col = tid; col < cols; col += blockDim.x) {
            int idx = row * cols + col;
            output[idx] = (input[idx] / rms) * weight[col];
        }
    }
}

// Optimized CUDA kernel using float4 for vectorized memory access
__global__ void rms_norm_matrix_kernel_float4(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    extern __shared__ float sdata[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each block processes one row, with multiple threads per row
    if (row < rows) {
        float thread_sum_squared = 0.0f;
        
        // Number of float4 elements per row
        int cols_float4 = cols / 4;
        
        // Each thread processes 4 elements at a time using float4
        for (int col_block = tid; col_block < cols_float4; col_block += blockDim.x) {
            // Convert to float4 pointers for vectorized memory access
            float4 input_vec = reinterpret_cast<float4*>(&input[row * cols])[col_block];
            
            // Calculate sum of squares for each component
            thread_sum_squared += input_vec.x * input_vec.x;
            thread_sum_squared += input_vec.y * input_vec.y;
            thread_sum_squared += input_vec.z * input_vec.z;
            thread_sum_squared += input_vec.w * input_vec.w;
        }
        
        // Handle remaining elements (if cols is not a multiple of 4)
        int remaining_start = cols_float4 * 4;
        for (int col = remaining_start + tid; col < cols; col += blockDim.x) {
            float val = input[row * cols + col];
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
            sdata[0] = rms; // Store rms in shared memory for other threads to use
        }
        __syncthreads();
        
        // All threads read the rms value
        rms = sdata[0];
        
        // Process float4 elements for output
        for (int col_block = tid; col_block < cols_float4; col_block += blockDim.x) {
            int col_base = col_block * 4;
            
            // Load input vector
            float4 input_vec = reinterpret_cast<float4*>(&input[row * cols])[col_block];
            
            // Load 4 consecutive weight values
            float w0 = weight[col_base];
            float w1 = weight[col_base + 1];
            float w2 = weight[col_base + 2];
            float w3 = weight[col_base + 3];
            
            // Perform normalization and weighting
            float4 output_vec;
            output_vec.x = (input_vec.x / rms) * w0;
            output_vec.y = (input_vec.y / rms) * w1;
            output_vec.z = (input_vec.z / rms) * w2;
            output_vec.w = (input_vec.w / rms) * w3;
            
            // Store the result
            reinterpret_cast<float4*>(&output[row * cols])[col_block] = output_vec;
        }
        
        // Handle remaining elements (if cols is not a multiple of 4)
        for (int col = remaining_start + tid; col < cols; col += blockDim.x) {
            int idx = row * cols + col;
            output[idx] = (input[idx] / rms) * weight[col];
        }
    }
}

// Check if the pointer is aligned for float4 operations (16-byte alignment)
bool is_aligned_float4(const void* ptr) {
    return ((uintptr_t)ptr % 16) == 0;
}

void rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    // Device memory pointers
    float *d_input, *d_weight, *d_output;
    
    // Calculate total memory size
    size_t input_size = rows * cols * sizeof(float);
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
    int blocksPerGrid = rows; // One block per row
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
        rms_norm_matrix_kernel_float4<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_input, d_weight, d_output, rows, cols, epsilon);
    } else {
        rms_norm_matrix_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_input, d_weight, d_output, rows, cols, epsilon);
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
