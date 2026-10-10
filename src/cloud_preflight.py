"""Run non-experimental environment checks before a cloud GPU job."""

import importlib
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GIB = 1024**3
NETWORK_URLS = ("https://huggingface.co", "https://hf-mirror.com")


def _import_module(module_name: str, display_name: str):
    """Import a dependency and print its version or a concise error."""
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        print(f"{display_name} version: UNAVAILABLE ({type(error).__name__}: {error})")
        return None

    print(f"{display_name} version: {getattr(module, '__version__', 'unknown')}")
    return module


def _network_check(url: str) -> None:
    """Check a URL without downloading models or datasets."""
    request = Request(url, method="HEAD", headers={"User-Agent": "CVIP-preflight"})
    try:
        with urlopen(request, timeout=10) as response:
            if 200 <= response.status < 400:
                print(f"Network {url}: OK")
            else:
                print(f"Network {url}: FAILED (HTTP {response.status})")
    except HTTPError as error:
        print(f"Network {url}: FAILED (HTTP {error.code})")
    except URLError as error:
        print(f"Network {url}: FAILED ({error.reason})")
    except TimeoutError:
        print(f"Network {url}: FAILED (timed out after 10s)")
    except Exception as error:
        print(f"Network {url}: FAILED ({type(error).__name__}: {error})")


def main() -> None:
    print(f"Python version: {sys.version}")

    torch = _import_module("torch", "PyTorch")
    torch_ok = torch is not None
    cuda_available = False
    gpu_name_ok = False
    if torch is not None:
        print(f"torch.version.cuda: {torch.version.cuda}")
        try:
            cuda_available = torch.cuda.is_available()
            print(f"torch.cuda.is_available(): {cuda_available}")
        except Exception as error:
            print(
                "torch.cuda.is_available(): FAILED "
                f"({type(error).__name__}: {error})"
            )

        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / GIB
                print(f"GPU name: {gpu_name}")
                print(f"GPU total memory: {total_memory_gb:.2f} GB")
                gpu_name_ok = bool(gpu_name)
            except Exception as error:
                print(f"GPU information: FAILED ({type(error).__name__}: {error})")
    else:
        print("PyTorch version: UNAVAILABLE")
        print("torch.version.cuda: UNAVAILABLE")
        print("torch.cuda.is_available(): False")

    transformers = _import_module("transformers", "transformers")
    datasets = _import_module("datasets", "datasets")
    pyarrow = _import_module("pyarrow", "pyarrow")

    working_directory = Path.cwd()
    print(f"Current working directory: {working_directory}")
    try:
        free_space_gb = shutil.disk_usage(working_directory).free / GIB
        print(f"Available disk space: {free_space_gb:.2f} GB")
    except OSError as error:
        print(f"Available disk space: UNAVAILABLE ({type(error).__name__}: {error})")

    for url in NETWORK_URLS:
        _network_check(url)

    core_ready = all(
        (
            torch_ok,
            cuda_available,
            gpu_name_ok,
            transformers is not None,
            datasets is not None,
            pyarrow is not None,
        )
    )
    print("PRECHECK PASSED" if core_ready else "PRECHECK FAILED")


if __name__ == "__main__":
    main()
