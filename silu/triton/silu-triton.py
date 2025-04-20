import torch
import triton 
import triton.language as tl
import triton.testing
import matplotlib.pyplot as plt
import numpy as np
import time
def naive_pytorch(x):
  return x * 1 / (1 + torch.exp(-x))

def main(args):
  torch.manual_seed(0)
  N = 8192
  D = 8192
  x = torch.rand(N, D, device="cuda")

  kernel = silu_naive_non_strided if args.naive else silu_non_strided

  if args.benchmark:
    results = benchmark_block_sizes(kernel,x, [8, 16, 32, 64, 128, 256])
    # Print summary
    print("\nSummary:")
    print("Block Size | Time (ms) | Throughput (GB/s) | Num Blocks")
    print("-" * 55)
    for r in results:
      print(f"{r['block_size']:^10d} | {r['avg_time_ms']:^9.3f} | {r['throughput_gb_s']:^15.2f}")
    return
  
  if args.triton_benchmark:
    benchmark.run(show_plots=True, print_data=True)
    return

  if args.single_benchmark:
    single_benchmark(x)
    return

  if not args.profile:
    output_torch = naive_pytorch(x)
    output_triton = kernel(x)
    is_close = torch.isclose(output_torch, output_triton)
    print(is_close.all(), is_close.sum())
    return

  with torch.profiler.profile(with_stack=True, profile_memory=True, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    # warmup  
    for _ in range(10):
      kernel(x)

    # profile
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    timings = []
    checksum = 0.0
    for _ in range(500):
      start_event.record()
      y = kernel(x)
      checksum += y.sum().item()
      end_event.record()
      torch.cuda.synchronize()
      timings.append(start_event.elapsed_time(end_event))

    average_time = sum(timings) / len(timings)

    total_data_processed = x.element_size() * x.numel() * 2 # read and write
    print(f"Average time: {average_time} ms")
    print(f"Elements: {x.numel():,}")
    print(f"Bytes per element: {x.element_size()}")
    print(f"Total data processed: {total_data_processed/1e9:.2f} GB")
    print(f"Memory throughput: {total_data_processed/average_time*1000/1e9:.2f} GB/s")
  prof.export_chrome_trace(f"silu-triton_trace.json")
  print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
  print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names=["N"],
    x_vals=[128 * i for i in range(2, 100)],
    plot_name="silu-triton-benchmark",
    line_arg='provider',
    line_vals=['triton', 'torch'],
    line_names=["Triton", "Torch"],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='GB/s',
    xlabel='N',
    args={'M': 4096}
  )
)
def benchmark(M, N, provider):
  device = torch.device("cuda")
  x = torch.rand(N, M, device=device)
  if provider == 'torch':
    ms = triton.testing.do_bench(lambda: naive_pytorch(x))
  if provider == 'triton':
    ms = triton.testing.do_bench(lambda: silu_non_strided(x))
  gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
  return gbps(ms) 

@triton.jit
def silu_kernel_naive(
  input_ptr,
  input_row_stride,
  output_ptr, 
  n_rows,
  n_cols: tl.constexpr,
):
  row_start = tl.program_id(0) * input_row_stride

  offsets = row_start + tl.arange(0, n_cols)
  mask = tl.arange(0, n_cols) < n_cols

  x = tl.load(input_ptr + offsets, mask=mask)
  output = x * tl.sigmoid(x)
  tl.store(output_ptr + offsets, output, mask=mask)

def silu_naive_non_strided(x, block_size=64, num_warps=2):
  output = torch.empty_like(x)
  n_rows, n_cols = x.shape

  grid = (n_rows,)
  silu_kernel_naive[grid](x, x.stride(0), output, n_rows, n_cols, num_stages=1, num_warps=num_warps)
  return output



@triton.jit
def silu_kernel_non_strided(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  m_start = pid_m * BLOCK_SIZE_M
  n_start = pid_n * BLOCK_SIZE_N

  m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
  n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)

  mask_m = m_offsets < n_rows
  mask_n = n_offsets < n_cols

  mask = mask_m[:,None] & mask_n[None,:]

  row_indices = m_offsets[:,None]
  col_indices = n_offsets[None, :]

  indicies = row_indices*n_cols+ col_indices

  x = tl.load(x_ptr + indicies, mask=mask)

  output = x * tl.sigmoid(x)

  tl.store(output_ptr + indicies, output, mask=mask)

def silu_non_strided(x, block_size=64, num_warps=2):
  output = torch.empty_like(x)
  n_rows, n_cols = x.shape

  grid = (triton.cdiv(n_rows, block_size), triton.cdiv(n_cols, block_size))
  silu_kernel_non_strided[grid](x, output, n_rows, n_cols, block_size, block_size, num_stages=1,num_warps=num_warps)
  return output

def benchmark_block_sizes(kernel, x, block_sizes=[8, 16, 32, 64, 128], warp_counts=[1, 2, 4, 8, 16, 32]):
    results = []
    
    # Prepare data structure for heatmap
    throughput_data = np.zeros((len(block_sizes), len(warp_counts)))
    
    # Loop through configurations
    for i, block_size in enumerate(block_sizes):
        for j, num_warps in enumerate(warp_counts):
            # Warmup
            for _ in range(10):
                y = kernel(x, block_size, num_warps)
            
            # Benchmark
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            timings = []
            
            for _ in range(100):
                start_event.record()
                y = kernel(x, block_size, num_warps)
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))
            
            # Calculate metrics
            avg_time = sum(timings) / len(timings)
            total_data = x.element_size() * x.numel() * 2  # read and write
            throughput = total_data / avg_time * 1000 / 1e9  # GB/s
            
            # Store result
            results.append({
                'block_size': block_size,
                'num_warps': num_warps,
                'avg_time_ms': avg_time,
                'throughput_gb_s': throughput
            })
            
            # Store in heatmap data
            throughput_data[i, j] = throughput
            
            print(f"Block Size: {block_size}x{block_size}, Warps: {num_warps}")
            print(f"  Avg time: {avg_time:.3f} ms")
            print(f"  Throughput: {throughput:.2f} GB/s")
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(throughput_data, cmap='viridis')
    plt.colorbar(im, label='Throughput (GB/s)')
    plt.title('Throughput vs. Block Size and Warp Count')
    
    # Set x and y ticks
    plt.xticks(np.arange(len(warp_counts)), warp_counts)
    plt.yticks(np.arange(len(block_sizes)), block_sizes)
    
    plt.xlabel('Number of Warps')
    plt.ylabel('Block Size')
    
    # Annotate heatmap with values
    for i in range(len(block_sizes)):
        for j in range(len(warp_counts)):
            text = plt.text(j, i, f"{throughput_data[i, j]:.1f}",
                           ha="center", va="center", color="w" if throughput_data[i, j] < throughput_data.max()*0.7 else "black")
    
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'silu_throughput_heatmap_{timestamp}.png')
    plt.show()
    
    return results

def single_benchmark(x, block_size=128, num_warps=32):
  # Warmup
  for _ in range(10):
      y = silu_non_strided(x, block_size, num_warps)
  
  # Benchmark
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  timings = []
  
  for _ in range(100):
      start_event.record()
      y = silu_non_strided(x, block_size, num_warps)
      end_event.record()
      torch.cuda.synchronize()
      timings.append(start_event.elapsed_time(end_event))
  
  # Calculate metrics
  avg_time = sum(timings) / len(timings)
  total_data = x.element_size() * x.numel() * 2  # read and write
  throughput = total_data / avg_time * 1000 / 1e9  # GB/s
  
  print(f"Block Size: {block_size}x{block_size}, Warps: {num_warps}")
  print(f"  Avg time: {avg_time:.3f} ms")
  print(f"  Throughput: {throughput:.2f} GB/s")

def plot_comparison_grid(torch_output, triton_output, grid_size=20):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get the shape
    n_rows, n_cols = torch_output.shape
    
    # Calculate how many elements to group together
    row_step = max(1, n_rows // grid_size)
    col_step = max(1, n_cols // grid_size)
    
    # Create a grid to store match/mismatch information
    match_grid = np.zeros((grid_size, grid_size))
    
    # For each cell in our visualization grid, check if the tensors match
    for i in range(grid_size):
        for j in range(grid_size):
            # Get row and column ranges for this cell
            row_start = i * row_step
            row_end = min(n_rows, (i + 1) * row_step)
            col_start = j * col_step
            col_end = min(n_cols, (j + 1) * col_step)
            
            # Extract the submatrix
            torch_sub = torch_output[row_start:row_end, col_start:col_end]
            triton_sub = triton_output[row_start:row_end, col_start:col_end]
            
            # Calculate the percentage of elements that match in this region
            is_close = torch.isclose(torch_sub, triton_sub, rtol=1e-5, atol=1e-8)
            match_percentage = is_close.float().mean().item() * 100
            match_grid[i, j] = match_percentage
    
    # Plot the grid
    plt.figure(figsize=(10, 8))
    im = plt.imshow(match_grid, cmap='RdYlGn', vmin=0, vmax=100)
    plt.colorbar(im, label='Percentage of matching values')
    plt.title('Comparison between PyTorch and Triton SiLU implementations')
    plt.xlabel(f'Columns (grouped by {col_step})')
    plt.ylabel(f'Rows (grouped by {row_step})')
    
    # Annotate regions with low match percentage
    for i in range(grid_size):
        for j in range(grid_size):
            if match_grid[i, j] < 99.9:  # Highlight only problematic areas
                plt.text(j, i, f'{match_grid[i, j]:.1f}%', 
                         ha='center', va='center', 
                         color='black' if match_grid[i, j] > 50 else 'white')
    
    plt.savefig(f'silu_comparison_grid_{int(time.time())}.png')
    plt.close()
    
    # Also plot a histogram of differences
    plt.figure(figsize=(10, 6))
    diff = (torch_output - triton_output).abs().flatten()
    diff_cpu = diff.cpu().numpy()
    plt.hist(diff_cpu, bins=50, log=True)
    plt.title('Distribution of absolute differences')
    plt.xlabel('Absolute difference')
    plt.ylabel('Count (log scale)')
    plt.savefig('silu_diff_histogram.png')
    plt.close()
    
    print(f"Maximum difference: {diff.max().item()}")
    print(f"Mean difference: {diff.mean().item()}")
    print(f"Images saved to silu_comparison_grid.png and silu_diff_histogram.png")

# # Add this at the end of your script
# plot_comparison_grid(output_torch, output_triton)
   
   
if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--profile", action="store_true")
  parser.add_argument("--benchmark", action="store_true")
  parser.add_argument("--triton_benchmark", action="store_true")
  parser.add_argument("--single_benchmark", action="store_true")
  parser.add_argument("--naive", action="store_true")
  args = parser.parse_args()
  main(args)