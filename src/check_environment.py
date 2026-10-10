"""Print basic information about the current Python and PyTorch environment."""

import os
import platform
import sys

import torch


def main():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Operating system: {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.machine()}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Project working directory: {os.getcwd()}")


if __name__ == "__main__":
    main()
