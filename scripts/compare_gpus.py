#!/usr/bin/env python3
"""Compare GPU results."""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.comparison import compare

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["h100", "a100"], default="h100")
    args = parser.parse_args()
    compare("MI300X", [])

if __name__ == "__main__":
    main()
