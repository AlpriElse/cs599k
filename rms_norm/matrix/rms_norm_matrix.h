#pragma once

// Main RMS normalization function
void rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon);

// Flag to control which implementation to use (0=auto, 1=original, 2=float4)
extern int rms_norm_implementation;