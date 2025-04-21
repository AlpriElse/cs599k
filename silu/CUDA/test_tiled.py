import silu
import numpy as np
import time

def main():
    # Test dimensions
    rows, cols = 1024, 1024
    
    # Create input and output arrays
    input_array_2d = np.random.randn(rows, cols).astype(np.float32)
    output_array_2d = np.zeros_like(input_array_2d)
    
    # Also create 1D arrays for comparison
    input_array_1d = input_array_2d.flatten()
    output_array_1d = np.zeros_like(input_array_1d)
    
    # Time the tiled version
    start = time.time()
    silu.py_silu_tiled(input_array_2d, output_array_2d)
    tiled_time = time.time() - start
    print(f"Tiled implementation took {tiled_time:.6f} seconds")
    
    # Time the non-tiled version
    start = time.time()
    silu.py_silu(input_array_1d, output_array_1d)
    non_tiled_time = time.time() - start
    print(f"Non-tiled implementation took {non_tiled_time:.6f} seconds")
    
    # Reshape the 1D result for comparison
    output_array_1d_reshaped = output_array_1d.reshape(rows, cols)
    
    # Verify results are similar
    max_diff = np.max(np.abs(output_array_2d - output_array_1d_reshaped))
    print(f"Maximum difference between implementations: {max_diff}")
    
    # Calculate speedup
    speedup = non_tiled_time / tiled_time
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main() 