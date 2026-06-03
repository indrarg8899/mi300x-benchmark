from setuptools import setup, find_packages
setup(
    name="mi300x-benchmark",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["torch>=2.3", "pyyaml"],
)
