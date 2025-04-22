#ifndef SILU_H
#define SILU_H

// Function to apply SiLU activation to a matrix
// input: pointer to input matrix in device memory
// output: pointer to output matrix in device memory
// matrix_size: size of square matrix (matrix_size x matrix_size)
// block_size: CUDA block size to use
// kernel_type: 0 for original, 1 for 2D grid, 2 for vector implementation
void silu(float *input, float *output, int matrix_size, int block_size, int kernel_type = 0);

#endif // SILU_H