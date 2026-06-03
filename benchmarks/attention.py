#!/usr/bin/env python3
"""Flash Attention benchmark."""
import torch, time

def bench(batch=8, heads=32, seq=2046, dim=128, iters=100, device="cuda:0"):
    q = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device=device)
    k = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device=device)
    v = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device=device)
    for _ in range(10):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    flops = 4 * batch * heads * seq * seq * dim
    tflops = flops / elapsed / 1e12
    print(f"Flash Attn seq={seq}: {tflops:.1f} TFLOPS ({elapsed*1000:.2f} ms)")
    return tflops

if __name__ == "__main__":
    bench()
