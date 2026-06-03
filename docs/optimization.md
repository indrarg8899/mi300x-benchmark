# MI300X Optimization Guide

## Memory
- Use `torch.channels_last` for convolutions
- Enable `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`

## Compute
- Prefer FP16/BF16 over FP32 (4x throughput)
- Use FP8 for inference (2x over FP16)
- Enable TF32 for FP32 GEMM

## Kernel
- Use `torch.compile()` for kernel fusion
- Flash Attention via `torch.nn.functional.scaled_dot_product_attention`
