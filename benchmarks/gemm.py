#!/usr/bin/env python3
"""GEMM benchmark."""
import torch, time, argparse

def bench(M=8192, N=8192, K=8192, dtype=torch.float16, iters=100, device="cuda:0"):
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    tflops = 2 * M * N * K / elapsed / 1e12
    print(f"GEMM {M}x{N}x{K} {dtype}: {tflops:.1f} TFLOPS ({elapsed*1000:.2f} ms)")
    return tflops

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    args = parser.parse_args()
    bench(args.size, args.size, args.size)
