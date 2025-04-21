import torch
import time
import silu_cuda

def benchmark():
    # Set up data
    torch.manual_seed(0)
    N = 8192
    D = 8192
    x = torch.rand(N, D, device="cuda")
    output = torch.zeros_like(x)
    
    # Warm-up runs
    for _ in range(10):
        silu_cuda.silu(x, output)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Timed runs
    start = time.time()
    for _ in range(100):
        silu_cuda.silu(x, output)
        torch.cuda.synchronize()  # Ensure kernel completion
    end = time.time()
    
    print(f"Average time: {(end - start) / 100 * 1000:.3f} ms")

if __name__ == "__main__":
    benchmark()