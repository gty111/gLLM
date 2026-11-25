import io
import os
import sys
from typing import List

from setuptools import find_packages, setup
from setuptools_rust import Binding, RustBin, RustExtension

ROOT_DIR = os.path.dirname(__file__)

assert sys.platform.startswith(
    "linux"
), "gLLM only supports Linux platform (including WSL)."


# rust/frontend (and the vLLM crates it uses) needs this toolchain; an older
# cargo fails deep inside a dependency, so refuse early with the fix.
MIN_RUST = (1, 95)


def check_rust_toolchain() -> None:
    import re
    import shutil
    import subprocess

    cargo = shutil.which("cargo")
    if cargo is None:
        return  # setuptools-rust reports a missing cargo itself
    try:
        out = subprocess.run([cargo, "--version"], capture_output=True, text=True).stdout
    except OSError:
        return
    m = re.search(r"(\d+)\.(\d+)", out)
    if m and (int(m[1]), int(m[2])) < MIN_RUST:
        raise SystemExit(
            f"gLLM needs Rust >= {MIN_RUST[0]}.{MIN_RUST[1]} to build gllm-frontend, "
            f"found `{out.strip()}` at {cargo}. Install it with "
            f"`rustup toolchain install {MIN_RUST[0]}.{MIN_RUST[1]}` and put that "
            "toolchain first on PATH."
        )


check_rust_toolchain()


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(p, "r", encoding="utf-8").read()
    return ""


def get_requirements() -> List[str]:
    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved = []
        for line in requirements:
            if line.startswith("-r "):
                resolved += _read_requirements(line.split()[1])
            else:
                resolved.append(line)
        return resolved

    return _read_requirements("requirements.txt")


setup(
    name="gllm",
    version="0.0.7",
    author="gtyinstinct",
    license="Apache 2.0",
    description="A high-throughput and memory-efficient inference and serving engine for LLMs",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=["gllm", *(f"gllm.{package}" for package in find_packages(where="python"))],
    package_dir={"gllm": "python"},
    python_requires=">=3.9",
    install_requires=get_requirements(),
    package_data={
        "gllm": [
            "layers/moe/fused_moe_triton/configs/*.json",
            "layers/ops/csrc/*.cu",
            "layers/quantization/csrc/marlin/*",
            "layers/quantization/csrc/marlin/include/sgl_kernel/*",
        ],
    },
    rust_extensions=[
        RustExtension(
            "gllm._cache_arena",
            path="rust/cache_arena/Cargo.toml",
            binding=Binding.PyO3,
            features=["pyo3/extension-module"],
            py_limited_api=True,
            debug=False,
        ),
        # The OpenAI-compatible HTTP front end (rust/frontend), installed as the
        # `gllm-frontend` executable next to the Python interpreter.
        RustBin(
            "gllm-frontend",
            path="rust/frontend/Cargo.toml",
            debug=False,
        ),
    ],
    zip_safe=False,
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
