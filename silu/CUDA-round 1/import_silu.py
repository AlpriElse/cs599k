import os
import sys
import torch

# Add PyTorch library path to LD_LIBRARY_PATH
torch_lib_path = torch.utils.cmake_prefix_path + '/lib'
os.environ['LD_LIBRARY_PATH'] = torch_lib_path + ':' + os.environ.get('LD_LIBRARY_PATH', '')

print(f"Set LD_LIBRARY_PATH to include: {torch_lib_path}")

# Try to import silu
try:
    import silu
    print("Successfully imported silu module")
except ImportError as e:
    print(f"Failed to import silu: {e}")
    sys.exit(1)

# If we get here, the import worked
print("silu module is ready to use")

# Export the module for others to use
__all__ = ['silu'] 