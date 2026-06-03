# MI300X Benchmark Suite

[![ROCm](https://img.shields.io/badge/ROCm-6.3+-red.svg)](https://rocm.docs.amd.com/)
[![MI300X](https://img.shields.io/badge/AMD-MI300X_192GB-blue.svg)](https://www.amd.com/en/products/accelerators/instinct/mi300x.html)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Comprehensive GPU benchmark suite for AMD MI300X (192GB HBM3, 5.3 TB/s).

## Features
- **Compute**: FP32/FP16/BF16/FP8/INT8 TFLOPS measurement
- **Memory**: HBM bandwidth, L2 cache, shared memory latency
- **Kernels**: GEMM, Flash Attention, All-Reduce, Conv2D, Reduction
- **Comparison**: MI300X vs H100 vs A100 results tables
- **Reports**: HTML/JSON/Markdown with charts
- **Profiling**: rocprof integration for kernel-level analysis

## Quick Start
```bash
pip install -r requirements.txt
python scripts/run_benchmark.py --suite full
python scripts/compare_gpus.py --baseline h100
```

## Benchmark Results (MI300X 192GB)

| Benchmark | MI300X | H100 80GB | Speedup |
|-----------|--------|-----------|---------|
| FP16 TFLOPS | 1,307 | 989 | 1.32x |
| FP8 TFLOPS | 2,614 | 1,979 | 1.32x |
| HBM BW (TB/s) | 5.3 | 3.35 | 1.58x |
| GEMM FP16 (TFLOPS) | 1,210 | 920 | 1.32x |
| Flash Attn (TFLOPS) | 850 | 650 | 1.31x |

## License
MIT
