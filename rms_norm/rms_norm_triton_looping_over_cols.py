import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr, output_ptr,
    stride, n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    row_idx = tl.program_id(0)
    
    # Compute row offset
    row_start = row_idx * stride
    
    # First pass: compute sum of squares for entire row
    sum_of_squares = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x_block = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        sum_of_squares += tl.sum(x_block * x_block, axis=0)
    
    # Compute RMS for entire row
    mean_of_squares = sum_of_squares / n_cols
    rms = tl.sqrt(mean_of_squares + eps)
    
    # Second pass: normalize and store
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x_block = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        x_normalized = x_block / rms
        tl.store(output_ptr + row_start + cols, x_normalized, mask=mask)

def triton_rms_norm(x, eps=1e-6):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    
    n_rows, n_cols = x.shape
    
    # Configure grid
    grid = (n_rows,)
    
    # Round up to nearest multiple of 128 for efficiency
    block_size = triton.next_power_of_2(n_cols)
    block_size = min(block_size, 1024)  # Max block size
    
    # Launch kernel
    rms_norm_kernel[grid](
        x, 
        output,
        x.stride(0), n_cols,
        eps, block_size
    )
    
    return output

torch.manual_seed(0)
D = 2**20
N = 512
device = torch.device("cuda")
x = torch.rand(N, D, device=device)

norm = torch.nn.RMSNorm(normalized_shape=D, device=device)

output_torch = norm(x)
output_triton = triton_rms_norm(x)
is_close = torch.isclose(output_torch, output_triton)
print(is_close.all())

# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)

# timings = []
# checksum = 0.0
# for i in range(100):
#     start_event.record()
#     y = triton_rms_norm(x)
#     checksum += y.sum(dim=-1) 
#     end_event.record()
#     torch.cuda.synchronize()
#     timings.append(start_event.elapsed_time(end_event))
# average_time = sum(timings) / len(timings)

# print(f"Average time: {average_time} ms")