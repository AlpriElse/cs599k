#pragma once

// Function to apply RMS normalization to a batch of vectors
void rms_norm_vector(float *input, float *weight, float *output, int cols, float epsilon);

// Flag to control which implementation to use (0=auto, 1=original, 2=float4)
extern int rms_norm_implementation;