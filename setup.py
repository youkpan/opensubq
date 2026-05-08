"""CRBSA: Codebook-Routed Block-Sparse Attention.

pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="crbsa",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2",
        "transformers>=4.40",
        "accelerate>=0.27",
        "safetensors",
    ],
    extras_require={
        "triton": ["triton>=2.2"],
        "eval": ["datasets", "numpy"],
        "dev": ["pytest", "ruff"],
    },
    description="Codebook-Routed Block-Sparse Attention for 1M+ Token Context",
    license="MIT",
)
