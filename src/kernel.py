"""Custom kernel benchmarks."""
import torch, time
from .benchmark import BenchmarkResult

class KernelBenchmark:
    def __init__(self, device, warmup=5, iterations=100):
        self.device = device
        self.warmup = warmup
        self.iterations = iterations
    
    def bench_gemm(self, M=4096, N=4096, K=4096, dtype=torch.float16):
        a = torch.randn(M, K, dtype=dtype, device=self.device)
        b = torch.randn(K, N, dtype=dtype, device=self.device)
        for _ in range(self.warmup):
            torch.mm(a, b)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(self.iterations):
            torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / self.iterations
        tflops = 2 * M * N * K / elapsed / 1e12
        return BenchmarkResult(f"gemm_{M}x{N}x{K}", str(self.device), tflops, "TFLOPS", elapsed)
    
    def bench_attention(self, batch=8, heads=32, seq=2046, dim=128):
        q = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device=self.device)
        k = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device=self.device)
        v = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device=self.device)
        for _ in range(self.warmup):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(self.iterations):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / self.iterations
        flops = 4 * batch * heads * seq * seq * dim
        tflops = flops / elapsed / 1e12
        return BenchmarkResult("flash_attention", str(self.device), tflops, "TFLOPS", elapsed)
    
    def run(self):
        results = [self.bench_gemm()]
        try:
            results.append(self.bench_attention())
        except Exception:
            pass
        return results
