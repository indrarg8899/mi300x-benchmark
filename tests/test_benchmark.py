"""Tests for benchmark suite."""
import unittest
from src.benchmark import BenchmarkResult, BenchmarkSuite
from src.comparison import REFERENCE_DATA

class TestBenchmarkResult(unittest.TestCase):
    def test_creation(self):
        r = BenchmarkResult("test", "cuda:0", 100.0, "TFLOPS")
        self.assertEqual(r.name, "test")
        self.assertEqual(r.value, 100.0)

class TestComparison(unittest.TestCase):
    def test_reference_data(self):
        self.assertIn("MI300X", REFERENCE_DATA)
        self.assertGreater(REFERENCE_DATA["MI300X"]["memory_gb"], 100)

if __name__ == "__main__":
    unittest.main()
