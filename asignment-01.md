
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

## RMS Norm

### TODO?
- OpenAI has a good visualization of Triton vs Pytorch kernels here
  - https://openai.com/index/triton/
- profiling in pytorch
  - https://medium.com/biased-algorithms/mastering-memory-profiling-in-pytorch-40007ced2e46


