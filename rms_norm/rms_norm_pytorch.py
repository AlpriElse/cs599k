import torch
import random

if not torch.cuda.is_available():
    raise ValueError("CUDA is not available")

seed = 241
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using GPU

N = 2**15
D = 256

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

device = torch.device("cuda")
x = torch.randn(N, D, device=device)

norm = torch.nn.RMSNorm(normalized_shape=D, device=device)

# warmup
for i in range(10):
    y = norm(x)

# benchmark
timings = []
checksum = 0.0
for i in range(100):
    start_event.record()
    y = norm(x)
    checksum += y.sum(dim=-1) 
    end_event.record()
    torch.cuda.synchronize()

    timings.append(start_event.elapsed_time(end_event))


average_time = sum(timings) / len(timings)

print(f"Average time: {average_time} ms")

# Calculate throughput
bytes_per_element = x.element_size()  # Usually 4 bytes for float32
time_in_seconds = average_time / 1000  # Convert ms to seconds

print(f"Elements: {x.numel():,}")
print(f"Bytes per element: {bytes_per_element}")