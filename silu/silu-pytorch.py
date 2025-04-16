import torch
import random

if not torch.cuda.is_available():
    raise ValueError("CUDA is not available")

seed = 241
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using GPU

N = 2**20
D = 256

def main(args):
    if args.manual_profile:
        manual_benchmark()
        return
    
    with torch.profiler.profile(with_stack=True, profile_memory=True, activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        device = torch.device("cuda")
        x = torch.randn(N, D, device=device)

        act = torch.nn.SiLU()

        # warmup
        for i in range(10):
            y = act(x)
        
        # benchmark
        for i in range(500):
            y = act(x)

    profile_name = "silu-pytorch"
    prof.export_chrome_trace(f"{profile_name}_trace.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    prof



def manual_benchmark():
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    device = torch.device("cuda")
    x = torch.randn(N, D, device=device)

    act = torch.nn.SiLU()

    # warmup
    for i in range(10):
        y = act(x)

    # benchmark
    timings = []
    checksum = 0.0
    for i in range(100):
        start_event.record()
        y = act(x)
        checksum += y.sum().item()
        end_event.record()
        torch.cuda.synchronize()

        timings.append(start_event.elapsed_time(end_event))

    print(f"Checksum: {checksum:.6f} (ensuring computation happened)")

    average_time = sum(timings) / len(timings)

    print(f"Average time: {average_time} ms")

    # Calculate throughput
    bytes_per_element = x.element_size()  # Usually 4 bytes for float32
    num_elements = x.numel() * 2  # Read + write operations
    time_in_seconds = average_time / 1000  # Convert ms to seconds
    throughput_gb_s = (bytes_per_element * num_elements) / time_in_seconds / 1e9

    print(f"Elements: {x.numel():,}")
    print(f"Bytes per element: {bytes_per_element}")
    print(f"Total data processed: {(bytes_per_element * num_elements)/1e9:.2f} GB")
    print(f"Memory throughput: {throughput_gb_s:.2f} GB/s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-profile", action="store_true")
    args = parser.parse_args()
    main(args)
