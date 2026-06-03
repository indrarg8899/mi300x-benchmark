"""rocprof integration."""
import subprocess, json, os

class ROCmProfiler:
    def __init__(self, output_dir="/tmp/rocprof"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def profile_script(self, script_path, args=None):
        cmd = ["rocprof", "--stats", "-o", f"{self.output_dir}/results.csv", "python", script_path]
        if args:
            cmd.extend(args)
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def parse_results(self):
        import csv
        csv_path = f"{self.output_dir}/results.csv"
        if not os.path.exists(csv_path):
            return None
        with open(csv_path) as f:
            return list(csv.DictReader(f))
