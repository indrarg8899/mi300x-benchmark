"""FP32/FP16/BF16 TFLOPS measurement."""
import torch, time
from .benchmark import BenchmarkResult

class ComputeBenchmark:
    def __init__(self, device, warmup=5, iterations=100):
        self.device = device
        self.warmup = warmup
        self.iterations = iterations
    
    def _bench_matmul(self, dtype, M=8192, N=8192, K=8192):
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
        return BenchmarkResult(f"matmul_{dtype}", str(self.device), tflops, "TFLOPS", elapsed)
    
    def run(self):
        results = []
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            try:
                results.append(self._bench_matmul(dtype))
            except Exception:
                pass
        return results
