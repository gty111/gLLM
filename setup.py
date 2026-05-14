import io
import os
import sys
from typing import List

from setuptools import find_packages, setup

ROOT_DIR = os.path.dirname(__file__)

assert sys.platform.startswith(
    "linux"
), "gLLM only supports Linux platform (including WSL)."


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
    version="0.1.0",
    author="gtyinstinct",
    license="Apache 2.0",
    description="A high-throughput and memory-efficient inference and serving engine for LLMs",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("benchmarks", "csrc", "docs", "examples", "tests*")),
    python_requires=">=3.9",
    install_requires=get_requirements(),
    package_data={
        "gllm": ["layers/moe/fused_moe_triton/configs/*.json"],
    },
)
