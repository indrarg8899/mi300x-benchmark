"""Main benchmark orchestrator."""
import time, json
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class BenchmarkResult:
    name: str
    device: str
    value: float
    unit: str
    duration_s: float = 0.0
    metadata: dict = field(default_factory=dict)

class BenchmarkSuite:
    def __init__(self, gpu_id=0, warmup=5, iterations=100):
        self.gpu_id = gpu_id
        self.warmup = warmup
        self.iterations = iterations
        self.results = []
    
    def run(self, suite="full"):
        import torch
        device = torch.device(f"cuda:{self.gpu_id}")
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}, Memory: {props.total_mem / 1e9:.1f} GB")
        
        from .compute import ComputeBenchmark
        from .memory import MemoryBenchmark
        from .kernel import KernelBenchmark
        
        if suite in ("full", "compute"):
            self.results.extend(ComputeBenchmark(device, self.warmup, self.iterations).run())
        if suite in ("full", "memory"):
            self.results.extend(MemoryBenchmark(device, self.warmup, self.iterations).run())
        if suite in ("full", "kernels"):
            self.results.extend(KernelBenchmark(device, self.warmup, self.iterations).run())
        return self.results
    
    def save(self, path):
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
