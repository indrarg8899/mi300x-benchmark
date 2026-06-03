# Benchmark Methodology

## Compute
- Matrix multiplication FP32/FP16/BF16 on square matrices
- Warmup: 5-10 iterations (discarded)
- Measurement: 100-200 iterations averaged
- TFLOPS = 2*M*N*K / elapsed_seconds / 1e12

## Memory
- Sequential read/write on contiguous GPU memory
- Sizes: 256MB to 16GB (HBM bandwidth)
- GB/s = size_bytes / elapsed_seconds / 1e9
