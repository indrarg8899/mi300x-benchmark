"""Shared utilities."""
import torch

def get_gpu_info(device=None):
    if device is None:
        device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "memory_gb": props.total_mem / 1e9,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
    }
