#include <cuda_runtime.h>
#include "rms_norm_vector.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <float.h>

// CPU version of RMS normalization for validation
void cpu_rms_norm_vector(float *input, float *weight, float *output, int batch_size, int cols, float epsilon) {
    for (int vec_idx = 0; vec_idx < batch_size; vec_idx++) {
        float sum_squared = 0.0f;
        
        // Compute sum of squares for this vector
        for (int i = 0; i < cols; i++) {
            float val = input[vec_idx * cols + i];
            sum_squared += val * val;
        }
        
        // Compute RMS normalization factor
        float rms = sqrt(sum_squared / (float)cols + epsilon);
        
        // Apply normalization and weight
        for (int i = 0; i < cols; i++) {
            int idx = vec_idx * cols + i;
            output[idx] = (input[idx] / rms) * weight[i];
        }
    }
}

// Function to display usage information
void display_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n\n", program_name);
    printf("Options:\n");
    printf("  --test                 Run validation against CPU implementation\n");
    printf("  --profile              Run performance profiling (10 and 100 iterations)\n");
    printf("  --impl=<number>        Select implementation:\n");
    printf("                           0: Auto-select (default)\n");
    printf("                           1: Original implementation\n");
    printf("                           2: Float4 vectorized implementation\n");
    printf("  --help                 Display this help message\n");
}

// Function to run the validation between CPU and GPU results
void run_validation(float *input, float *weight, float *output, int batch_size, int cols, float epsilon) {
    printf("Validating results...\n");
    
    // Allocate memory for CPU output
    float *cpu_output = (float*)malloc(batch_size * cols * sizeof(float));
    if (!cpu_output) {
        printf("Error: CPU output memory allocation failed\n");
        return;
    }
    
    // Measure CPU time for comparison
    clock_t cpu_start = clock();
    
    // Only compute CPU version on a small subset (first 10 vectors) to save time
    const int cpu_test_batch = 10;
    cpu_rms_norm_vector(input, weight, cpu_output, cpu_test_batch, cols, epsilon);
    
    clock_t cpu_end = clock();
    double cpu_elapsed_time = 1000.0 * (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    
    printf("CPU RMS normalization for %d vectors: %.2f ms\n", cpu_test_batch, cpu_elapsed_time);
    
    // Extrapolate CPU time for the full batch
    double estimated_full_cpu_time = cpu_elapsed_time * (batch_size / (double)cpu_test_batch);
    printf("Estimated CPU time for full batch: %.2f ms\n", estimated_full_cpu_time);
    
    // Check a few elements
    bool all_correct = true;
    float max_diff = 0.0f;
    
    for (int vec = 0; vec < cpu_test_batch; vec++) {
        for (int i = 0; i < 10; i++) {
            int idx = vec * cols + i;
            float diff = fabs(output[idx] - cpu_output[idx]);
            max_diff = fmax(max_diff, diff);
            
            if (diff > 1e-3f) {
                all_correct = false;
                printf("Mismatch at vector %d, element %d: GPU=%f, CPU=%f, diff=%f\n", 
                       vec, i, output[idx], cpu_output[idx], diff);
                break;
            }
        }
        if (!all_correct) break;
    }
    
    if (all_correct) {
        printf("Validation PASSED! Max difference: %e\n", max_diff);
    } else {
        printf("Validation FAILED!\n");
    }
    
    free(cpu_output);
}

// Function to run a single kernel execution and return timing
float run_kernel(float *input, float *weight, float *output, int cols, float epsilon) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    rms_norm_vector(input, weight, output, cols, epsilon);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return elapsed_time;
}

// Function to run profiling with multiple iterations
void run_profiling(float *input, float *weight, float *output, int cols, float epsilon) {
    printf("Running profiling...\n");
    
    // First run: 10 iterations
    printf("Running 10 iterations...\n");
    float total_time_10 = 0.0f;
    for (int i = 0; i < 10; i++) {
        float time = run_kernel(input, weight, output, cols, epsilon);
        total_time_10 += time;
        printf("  Iteration %d: %.2f ms\n", i+1, time);
    }
    float avg_time_10 = total_time_10 / 10.0f;
    printf("Average time for 10 iterations: %.2f ms\n", avg_time_10);
    
    // Second run: 100 iterations
    printf("\nRunning 100 iterations...\n");
    float total_time_100 = 0.0f;
    float min_time = FLT_MAX;
    float max_time = 0.0f;
    
    for (int i = 0; i < 100; i++) {
        float time = run_kernel(input, weight, output, cols, epsilon);
        total_time_100 += time;
        min_time = fmin(min_time, time);
        max_time = fmax(max_time, time);
        
        if (i % 10 == 9) {
            printf("  Completed %d iterations...\n", i+1);
        }
    }
    
    float avg_time_100 = total_time_100 / 100.0f;
    printf("Average time for 100 iterations: %.2f ms\n", avg_time_100);
    printf("Min time: %.2f ms, Max time: %.2f ms, Range: %.2f ms\n", 
           min_time, max_time, max_time - min_time);
}

int main(int argc, char *argv[]) {
    bool run_test = false;
    bool run_profile = false;
    bool show_help = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0) {
            run_test = true;
        } else if (strcmp(argv[i], "--profile") == 0) {
            run_profile = true;
        } else if (strncmp(argv[i], "--impl=", 7) == 0) {
            // Set implementation flag
            rms_norm_implementation = atoi(argv[i] + 7);
        } else if (strcmp(argv[i], "--help") == 0) {
            show_help = true;
        } else {
            printf("Unknown argument: %s\n", argv[i]);
            show_help = true;
        }
    }
    
    if (show_help) {
        display_usage(argv[0]);
        return 0;
    }
    
    // Set fixed random seed
    srand(42);
    
    // Vector dimensions: (1, batch_size, cols)
    const int batch_size = 1024;
    const int cols = 1024;
    const float epsilon = 1e-5f;
    
    printf("Initializing tensor of shape (1,%d,%d)...\n", batch_size, cols);
    printf("Using implementation mode: %d\n", rms_norm_implementation);
    
    // Allocate host memory
    size_t input_size = batch_size * cols * sizeof(float);
    float *input = (float*)malloc(input_size);
    float *weight = (float*)malloc(cols * sizeof(float));
    float *output = (float*)malloc(input_size);
    
    if (!input || !weight || !output) {
        printf("Error: Memory allocation failed\n");
        return 1;
    }
    
    // Initialize input matrix with random values
    for (int i = 0; i < batch_size * cols; i++) {
        input[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // Random values between -1 and 1
    }
    
    // Initialize weight vector with random values
    for (int i = 0; i < cols; i++) {
        weight[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // Random values between -1 and 1
    }
    
    // Default mode: run once and time it
    if (!run_test && !run_profile) {
        printf("Running RMS normalization once...\n");
        float elapsed_time = run_kernel(input, weight, output, cols, epsilon);
        printf("GPU RMS normalization complete! Time: %.2f ms\n", elapsed_time);
    }
    
    // Test mode: validate against CPU
    if (run_test) {
        printf("Running RMS normalization for testing...\n");
        float elapsed_time = run_kernel(input, weight, output, cols, epsilon);
        printf("GPU RMS normalization complete! Time: %.2f ms\n", elapsed_time);
        
        run_validation(input, weight, output, batch_size, cols, epsilon);
    }
    
    // Profile mode: run multiple iterations
    if (run_profile) {
        run_profiling(input, weight, output, cols, epsilon);
    }
    
    // Free host memory
    free(input);
    free(weight);
    free(output);
    
    return 0;
}