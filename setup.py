"""Setup script for FoloVLLM."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="folovllm",
    version="0.1.0",
    author="FoloVLLM Contributors",
    description="A Lightweight LLM Inference Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/folovllm",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "reference"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "flash": [
            "flash-attn>=2.0.0",
        ],
        "quantization": [
            "auto-gptq>=0.6.0",
            "optimum>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "folovllm=folovllm.cli:main",
        ],
    },
)

