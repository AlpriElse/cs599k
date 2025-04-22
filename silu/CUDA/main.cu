#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <random>
#include <string>
#include <chrono>
#include <vector>
#include "silu.h"

// Matrix dimensions
#define MATRIX_SIZE 8192
#define MATRIX_BYTES (MATRIX_SIZE * MATRIX_SIZE * sizeof(float))

// Default block size for kernel
#define BLOCK_SIZE 256

// Static seed for random number generation
static const unsigned int SEED = 12345;

// Number of profiling iterations
#define PROFILE_ITERATIONS 100

// Kernel types
const char* KERNEL_NAMES[] = {"Original Row-based", "2D Grid", "Vector-based"};

const std::vector<int> BLOCK_SIZES_TO_TEST = {32, 64, 128, 256, 512, 1024};

void run_profile(float* d_input, float* d_output, int block_size, int kernel_type = 0) {
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup phase - run a few iterations before timing
    const int WARMUP_ITERATIONS = 10;
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        silu(d_input, d_output, MATRIX_SIZE, block_size, kernel_type);
    }
    cudaDeviceSynchronize(); // Ensure warmup is complete
    
    // Start timing
    cudaEventRecord(start);
    
    // Run multiple iterations for profiling
    for (int i = 0; i < PROFILE_ITERATIONS; i++) {
        silu(d_input, d_output, MATRIX_SIZE, block_size, kernel_type);
    }
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate throughput
    double bytes_processed = 2* static_cast<double>(MATRIX_SIZE) * MATRIX_SIZE * PROFILE_ITERATIONS * sizeof(float);
    double seconds = milliseconds / 1000.0;
    double gb_per_sec = (bytes_processed / seconds) / (1024.0 * 1024.0 * 1024.0);
    
    // Print results for this block size
    if (kernel_type >= 0) {
        std::cout << "Kernel: " << KERNEL_NAMES[kernel_type] << ", Block size " << block_size << ":" << std::endl;
    } else {
        std::cout << "Block size " << block_size << ":" << std::endl;
    }
    std::cout << "  Total time for " << PROFILE_ITERATIONS << " iterations: " << milliseconds << " ms" << std::endl;
    std::cout << "  Average time per iteration: " << milliseconds / PROFILE_ITERATIONS << " ms" << std::endl;
    std::cout << "  Throughput: " << gb_per_sec << " GB/s" << std::endl;
    std::cout << std::endl;
    
    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    // Parse command line arguments
    bool profile_mode = false;
    bool profile_block_sizes = false;
    bool profile_kernels = false;
    int default_kernel = 2;  // Use vector-based kernel by default
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--profile") {
            profile_mode = true;
            std::cout << "Running in profile mode with " << PROFILE_ITERATIONS << " iterations" << std::endl;
        } else if (arg == "--profile_block_sizes") {
            profile_block_sizes = true;
            std::cout << "Running block size profiling with " << PROFILE_ITERATIONS << " iterations per block size" << std::endl;
        } else if (arg == "--profile_kernels") {
            profile_kernels = true;
            std::cout << "Running kernel implementation profiling with " << PROFILE_ITERATIONS << " iterations per kernel" << std::endl;
        } else if (arg == "--kernel" && i + 1 < argc) {
            default_kernel = std::stoi(argv[++i]);
            if (default_kernel < 0 || default_kernel > 2) {
                std::cerr << "Invalid kernel type: " << default_kernel << ". Using default (2)." << std::endl;
                default_kernel = 2;
            }
        }
    }
    
    std::cout << "Using kernel: " << KERNEL_NAMES[default_kernel] << std::endl;
    
    // Device memory pointers
    float *d_input = nullptr;
    float *d_output = nullptr;
    
    // Host memory pointers
    float *h_input = nullptr;
    float *h_output = nullptr;
    
    // Allocate host memory
    h_input = new float[MATRIX_SIZE * MATRIX_SIZE];
    h_output = new float[MATRIX_SIZE * MATRIX_SIZE];
    
    // Initialize random number generator with static seed
    std::mt19937 rng(SEED);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
    // Fill input matrix with random values
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_input[i] = dist(rng);
    }
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, MATRIX_BYTES);
    cudaMalloc((void**)&d_output, MATRIX_BYTES);
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, MATRIX_BYTES, cudaMemcpyHostToDevice);
    
    if (profile_kernels) {
        std::cout << "Profiling different kernel implementations:" << std::endl;
        std::cout << "Matrix size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
        std::cout << "Block size: " << BLOCK_SIZE << std::endl;
        std::cout << "Iterations per kernel: " << PROFILE_ITERATIONS << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Test each kernel implementation
        for (int kernel_type = 0; kernel_type < 3; kernel_type++) {
            run_profile(d_input, d_output, BLOCK_SIZE, kernel_type);
            
            // Verify results for each kernel
            cudaMemcpy(h_output, d_output, MATRIX_BYTES, cudaMemcpyDeviceToHost);
            
            std::cout << "Verification for " << KERNEL_NAMES[kernel_type] << ":" << std::endl;
            bool passed = true;
            for (int i = 0; i < 5; i++) {
                int idx = i * MATRIX_SIZE;
                float expected = h_input[idx] / (1.0f + expf(-h_input[idx]));
                float error = std::abs(h_output[idx] - expected) / expected;
                
                std::cout << "  Element[" << idx << "]: " << h_output[idx] << ", Expected: " << expected 
                          << ", Relative error: " << (error * 100.0f) << "%" << std::endl;
                
                if (error > 0.01f) { // 1% error tolerance
                    passed = false;
                }
            }
            std::cout << "  Verification " << (passed ? "PASSED" : "FAILED") << std::endl;
            std::cout << std::endl;
        }
    } else if (profile_block_sizes) {
        std::cout << "Profiling different block sizes:" << std::endl;
        std::cout << "Matrix size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
        std::cout << "Iterations per block size: " << PROFILE_ITERATIONS << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        for (int block_size : BLOCK_SIZES_TO_TEST) {
            run_profile(d_input, d_output, block_size);
        }
    } else if (profile_mode) {
        run_profile(d_input, d_output, BLOCK_SIZE, default_kernel);
    } else {
        // Regular mode - just run once
        printf("val %f", h_input[0]);
        
        silu(d_input, d_output, MATRIX_SIZE, BLOCK_SIZE, default_kernel);
    }
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, MATRIX_BYTES, cudaMemcpyDeviceToHost);
    
    if (!profile_mode && !profile_block_sizes && !profile_kernels) {
        // Verify results (sample check) - only in regular mode
        std::cout << "Verifying SiLU operation on sample elements:" << std::endl;
        for (int i = 0; i < 5; i++) {
            int idx = i * MATRIX_SIZE;
            float expected = h_input[idx] / (1.0f + expf(-h_input[idx]));
            std::cout << "Input[" << idx << "] = " << h_input[idx] 
                      << ", Output[" << idx << "] = " << h_output[idx]
                      << ", Expected ≈ " << expected << std::endl;
        }
    }
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Free host memory
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}