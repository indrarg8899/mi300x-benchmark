"""HBM bandwidth and latency tests."""
import torch, time
from .benchmark import BenchmarkResult

class MemoryBenchmark:
    def __init__(self, device, warmup=5, iterations=100):
        self.device = device
        self.warmup = warmup
        self.iterations = iterations
    
    def _bench_bandwidth(self, size_bytes, direction="read"):
        size_elems = size_bytes // 4
        src = torch.randn(size_elems, device=self.device)
        dst = torch.empty_like(src)
        for _ in range(self.warmup):
            dst.copy_(src)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(self.iterations):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / self.iterations
        bw_gbs = size_bytes / elapsed / 1e9
        return BenchmarkResult(f"bw_{direction}_{size_bytes//1024//1024}MB", str(self.device), bw_gbs, "GB/s", elapsed)
    
    def run(self):
        results = []
        for size in [256*1024*1024, 1024*1024*1024, 4*1024*1024*1024, 16*1024*1024*1024]:
            try:
                results.append(self._bench_bandwidth(size, "read"))
                results.append(self._bench_bandwidth(size, "write"))
            except RuntimeError:
                pass
        return results
