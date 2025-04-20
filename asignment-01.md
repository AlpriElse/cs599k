Using ncu use `--set full
https://forums.developer.nvidia.com/t/some-metric-set-and-section-are-not-enable/266223


## Silu

### Profiling silu in pytorch
Average time: 0.6012162555456162 ms
Elements: 268,435,456
Bytes per element: 4
Total data processed: 2.15 GB
Memory throughput: 3571.90 GB/s

### Implement silu triton
- jesus, something about how I need to stride for non-contiguous tensors 
- going to implement non-strided version first

### Debugging timing
- https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
  - indeed need to sync after end event record

### Profiling silu in PyTorch (round 2)
- seems like I had some errors; had to 2x the throughput number due to account for both the read and write from cuda memory
- also had to do something/store the result of running otherwise I think the Pytorch compiler was doing an optimization and not doing the write operation

Average time: 0.9623593592643738 ms
Elements: 268,435,456
Bytes per element: 4
Total data processed: 2.15 GB
Memory throughput: 2231.48 GB/s


lol not sure how I got those numbers abov

### Pinning dim to 8192x8192 and experimenting with block sizes
```
ubuntu@192-222-51-232:~/pripri-labs/cs599k/silu$ python3 silu-triton.py --benchmark

Block Size: 8x8
Number of blocks: 1048576
Average time: 1.410 ms
Memory throughput: 380.77 GB/s

Block Size: 16x16
Number of blocks: 262144
Average time: 0.359 ms
Memory throughput: 1493.83 GB/s

Block Size: 32x32
Number of blocks: 65536
Average time: 0.322 ms
Memory throughput: 1669.81 GB/s

Block Size: 64x64
Number of blocks: 16384
Average time: 0.314 ms
Memory throughput: 1712.22 GB/s

Block Size: 128x128
Number of blocks: 4096
Average time: 0.317 ms
Memory throughput: 1693.38 GB/s

Summary:
Block Size | Time (ms) | Throughput (GB/s) | Num Blocks
-------------------------------------------------------
    8      |   1.410   |     380.77      |  1048576  
    16     |   0.359   |     1493.83     |   262144  
    32     |   0.322   |     1669.81     |   65536   
    64     |   0.314   |     1712.22     |   16384   
   128     |   0.317   |     1693.38     |    4096  
```

## RMS Norm

### TODO?
- OpenAI has a good visualization of Triton vs Pytorch kernels here
  - https://openai.com/index/triton/
- profiling in pytorch
  - https://medium.com/biased-algorithms/mastering-memory-profiling-in-pytorch-40007ced2e46

# Things to try next
- triton benchmarking / autotune 
- rms norm, splitting it into two separate kernels


### 2025-04-18
- Found this comment about what num_stages actually is
  - https://github.com/triton-lang/triton/discussions/512
  - mr. chatgpt helping:
    - software piplining of memory loads and stores, trade off of latency vs shared memory usage/"register pressure"
    - too many registers used in a Kernel OR too much Shared Memory Usage per thread block = fewer blocks can run concurrently  ---- there's limited registers and Shared Memory per block
    

### 2025-04-19
- CUDA + Pybind
  - https://github.com/torstem/demo-cuda-pybind11

