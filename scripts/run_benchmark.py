#!/usr/bin/env python3
"""Run benchmark suite."""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.benchmark import BenchmarkSuite
from src.report import generate_html, generate_markdown

def main():
    parser = argparse.ArgumentParser(description="MI300X Benchmark Suite")
    parser.add_argument("--suite", choices=["full", "compute", "memory", "kernels"], default="full")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default="./results")
    parser.add_argument("--format", choices=["html", "markdown", "json"], default="html")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    suite = BenchmarkSuite(args.gpu, args.warmup, args.iterations)
    results = suite.run(args.suite)
    
    if args.format == "html":
        generate_html(results, f"{args.output}/report.html")
    elif args.format == "markdown":
        generate_markdown(results, f"{args.output}/REPORT.md")
    suite.save(f"{args.output}/results.json")
    print(f"Results saved to {args.output}/")

if __name__ == "__main__":
    main()
