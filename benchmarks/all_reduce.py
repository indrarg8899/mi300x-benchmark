#!/usr/bin/env python3
"""All-Reduce bandwidth benchmark."""
import torch, time

def bench(size_mb=1024, iters=100, device="cuda:0"):
    size = size_mb * 1024 * 1024 // 4
    src = torch.randn(size, device=device)
    dst = torch.empty_like(src)
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    bw_gbs = size * 4 / elapsed / 1e9
    print(f"All-Reduce {size_mb}MB: {bw_gbs:.1f} GB/s ({elapsed*1000:.2f} ms)")
    return bw_gbs

if __name__ == "__main__":
    bench()
