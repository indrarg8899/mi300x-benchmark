"""Cross-GPU comparison."""
REFERENCE_DATA = {
    "MI300X": {"fp16_tflops": 1307, "fp8_tflops": 2614, "hbm_bw_tbs": 5.3, "memory_gb": 192},
    "H100": {"fp16_tflops": 989, "fp8_tflops": 1979, "hbm_bw_tbs": 3.35, "memory_gb": 80},
    "A100": {"fp16_tflops": 312, "fp8_tflops": 624, "hbm_bw_tbs": 2.0, "memory_gb": 80},
}

def compare(gpu_name, results):
    ref = REFERENCE_DATA.get(gpu_name, {})
    print(f"\n{gpu_name} vs Reference Data")
    for key, expected in ref.items():
        print(f"  {key}: {expected} (reference)")
