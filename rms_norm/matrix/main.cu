#include <cuda_runtime.h>
#include "rms_norm_matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <float.h>

// CPU version of RMS normalization for validation
void cpu_rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    for (int row = 0; row < rows; row++) {
        float sum_squared = 0.0f;
        
        // Compute sum of squares for this row
        for (int col = 0; col < cols; col++) {
            float val = input[row * cols + col];
            sum_squared += val * val;
        }
        
        // Compute RMS normalization factor - explicitly cast cols to float
        float rms = sqrt(sum_squared / (float)cols + epsilon);
        
        // Apply normalization and weight
        for (int col = 0; col < cols; col++) {
            int idx = row * cols + col;
            output[idx] = (input[idx] / rms) * weight[col];
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
void run_validation(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    printf("Validating results...\n");
    
    // Allocate memory for CPU output
    float *cpu_output = (float*)malloc(rows * cols * sizeof(float));
    if (!cpu_output) {
        printf("Error: CPU output memory allocation failed\n");
        return;
    }
    
    // Measure CPU time for comparison (only on a small subset)
    const int cpu_test_rows = 2;
    clock_t cpu_start = clock();
    
    // Only compute CPU version on a small subset (first 2 rows) to save time
    cpu_rms_norm_matrix(input, weight, cpu_output, cpu_test_rows, cols, epsilon);
    
    clock_t cpu_end = clock();
    double cpu_elapsed_time = 1000.0 * (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    
    printf("CPU RMS normalization for %d rows: %.2f ms\n", cpu_test_rows, cpu_elapsed_time);
    
    // Extrapolate CPU time for the full matrix
    double estimated_full_cpu_time = cpu_elapsed_time * (rows / (double)cpu_test_rows);
    printf("Estimated CPU time for full matrix: %.2f ms\n", estimated_full_cpu_time);
    
    // Check a few elements
    bool all_correct = true;
    float max_diff = 0.0f;
    
    for (int i = 0; i < cpu_test_rows; i++) {
        for (int j = 0; j < 10; j++) {
            int idx = i * cols + j;
            float diff = fabs(output[idx] - cpu_output[idx]);
            max_diff = fmax(max_diff, diff);
            
            if (diff > 1e-3f) {
                all_correct = false;
                printf("Mismatch at (%d,%d): GPU=%f, CPU=%f, diff=%f\n", i, j, output[idx], cpu_output[idx], diff);
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
float run_kernel(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    rms_norm_matrix(input, weight, output, rows, cols, epsilon);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return elapsed_time;
}

// Function to run profiling with multiple iterations
void run_profiling(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    printf("Running profiling...\n");
    
    // First run: 10 iterations
    printf("Running 10 iterations...\n");
    float total_time_10 = 0.0f;
    for (int i = 0; i < 10; i++) {
        float time = run_kernel(input, weight, output, rows, cols, epsilon);
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
        float time = run_kernel(input, weight, output, rows, cols, epsilon);
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
    
    // Matrix dimensions
    const int rows = 8192;
    const int cols = 8192;
    const float epsilon = 1e-5f;
    
    // Make cols a multiple of 4 to enable float4 optimization
    // const int cols = 8192 + (4 - (8192 % 4)) % 4;
    
    printf("Initializing %dx%d matrix...\n", rows, cols);
    printf("Using implementation mode: %d\n", rms_norm_implementation);
    
    // Allocate host memory
    float *input = (float*)malloc(rows * cols * sizeof(float));
    float *weight = (float*)malloc(cols * sizeof(float));
    float *output = (float*)malloc(rows * cols * sizeof(float));
    
    if (!input || !weight || !output) {
        printf("Error: Memory allocation failed\n");
        return 1;
    }
    
    // Initialize input matrix with random values - fix RAND_MAX conversion warning
    for (int i = 0; i < rows * cols; i++) {
        input[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // Random values between -1 and 1
    }
    
    // Initialize weight vector with random values - fix RAND_MAX conversion warning
    for (int i = 0; i < cols; i++) {
        weight[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // Random values between -1 and 1
    }
    
    // Default mode: run once and time it
    if (!run_test && !run_profile) {
        printf("Running RMS normalization once...\n");
        float elapsed_time = run_kernel(input, weight, output, rows, cols, epsilon);
        printf("GPU RMS normalization complete! Time: %.2f ms\n", elapsed_time);
    }
    
    // Test mode: validate against CPU
    if (run_test) {
        printf("Running RMS normalization for testing...\n");
        float elapsed_time = run_kernel(input, weight, output, rows, cols, epsilon);
        printf("GPU RMS normalization complete! Time: %.2f ms\n", elapsed_time);
        
        run_validation(input, weight, output, rows, cols, epsilon);
    }
    
    // Profile mode: run multiple iterations
    if (run_profile) {
        run_profiling(input, weight, output, rows, cols, epsilon);
    }
    
    // Free host memory
    free(input);
    free(weight);
    free(output);
    
    return 0;
}