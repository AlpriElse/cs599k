import silu
import numpy as np

# Create input and output arrays
input_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
output_array = np.zeros_like(input_array)

# Call py_silu with the numpy arrays
silu.py_silu(input_array, output_array)

# Now output_array should contain the results
print(output_array)