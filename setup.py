import importlib.util
import io
import logging
import os
import re
import shutil
import sys
from shutil import which
from typing import List

import torch
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

# gLLM only supports Linux platform
assert sys.platform.startswith(
    "linux"
), "gLLM only supports Linux platform (including WSL)."

MAIN_CUDA_VERSION = "12.1"


def is_sccache_available() -> bool:
    return which("sccache") is not None


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


class CMakeExtension(Extension):

    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=True, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class precompiled_build_ext(build_ext):
    """Disables extension building when using precompiled binaries."""

    def run(self) -> None:
        assert _is_cuda(), "VLLM_USE_PRECOMPILED is only supported for CUDA builds"

    def build_extensions(self) -> None:
        print("Skipping build_ext: using precompiled extensions.")
        return


class precompiled_wheel_utils:
    """Extracts libraries and other files from an existing wheel."""

    @staticmethod
    def extract_precompiled_and_patch_package(wheel_url_or_path: str) -> dict:
        import tempfile
        import zipfile

        temp_dir = None
        try:
            if not os.path.isfile(wheel_url_or_path):
                wheel_filename = wheel_url_or_path.split("/")[-1]
                temp_dir = tempfile.mkdtemp(prefix="vllm-wheels")
                wheel_path = os.path.join(temp_dir, wheel_filename)
                print(f"Downloading wheel from {wheel_url_or_path} to {wheel_path}")
                from urllib.request import urlretrieve

                urlretrieve(wheel_url_or_path, filename=wheel_path)
            else:
                wheel_path = wheel_url_or_path
                print(f"Using existing wheel at {wheel_path}")

            package_data_patch = {}

            with zipfile.ZipFile(wheel_path) as wheel:
                files_to_copy = [
                    "vllm/_C.abi3.so",
                    "vllm/_moe_C.abi3.so",
                    "vllm/_flashmla_C.abi3.so",
                    "vllm/_flashmla_extension_C.abi3.so",
                    "vllm/_sparse_flashmla_C.abi3.so",
                    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
                    "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
                    "vllm/cumem_allocator.abi3.so",
                ]

                compiled_regex = re.compile(
                    r"vllm/vllm_flash_attn/(?:[^/.][^/]*/)*(?!\.)[^/]*\.py"
                )
                file_members = list(
                    filter(lambda x: x.filename in files_to_copy, wheel.filelist)
                )
                file_members += list(
                    filter(lambda x: compiled_regex.match(x.filename), wheel.filelist)
                )

                for file in file_members:
                    target_file_name = file.filename.replace("vllm/", "gllm/")
                    print(f"[extract] {file.filename} to {target_file_name}")
                    target_path = os.path.join(".", target_file_name)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with (
                        wheel.open(file.filename) as src,
                        open(target_path, "wb") as dst,
                    ):
                        shutil.copyfileobj(src, dst)

                    pkg = os.path.dirname(file.filename).replace("/", ".")
                    package_data_patch.setdefault(pkg, []).append(
                        os.path.basename(file.filename)
                    )

            return package_data_patch
        finally:
            if temp_dir is not None:
                print(f"Removing temporary directory {temp_dir}")
                shutil.rmtree(temp_dir)


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return has_cuda


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_gllm_version() -> str:
    version = "0.0.4"
    version += "+precompiled"

    return version


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("requirements.txt")

    return requirements


ext_modules = []

ext_modules.append(CMakeExtension(name="gllm._C"))
ext_modules.append(CMakeExtension(name="gllm._moe_C"))
ext_modules.append(CMakeExtension(name="gllm.vllm_flash_attn._vllm_fa2_C"))
ext_modules.append(CMakeExtension(name="gllm.vllm_flash_attn._vllm_fa3_C"))

package_data = {
    "gllm": [
        "layers/moe/fused_moe_triton/configs/*.json",
    ]
}

assert _is_cuda(), "VLLM_USE_PRECOMPILED is only supported for CUDA builds"
wheel_location = os.getenv("GLLM_PRECOMPILED_WHEEL_LOCATION", None)
if wheel_location is not None:
    wheel_url = wheel_location
else:
    import platform

    arch = platform.machine()
    if arch == "x86_64":
        wheel_tag = "manylinux1_x86_64"
    elif arch == "aarch64":
        wheel_tag = "manylinux2014_aarch64"
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    wheel_url = "https://wheels.vllm.ai/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm-0.11.0%2Bcu129-cp38-abi3-manylinux1_x86_64.whl"

patch = precompiled_wheel_utils.extract_precompiled_and_patch_package(wheel_url)
for pkg, files in patch.items():
    package_data.setdefault(pkg, []).extend(files)

setup(
    name="gllm",
    version=get_gllm_version(),
    author="gtyinstinct",
    license="Apache 2.0",
    description=(
        "A high-throughput and memory-efficient inference and "
        "serving engine for LLMs"
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(
        exclude=("benchmarks", "csrc", "docs", "examples", "tests*")
    ),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": precompiled_build_ext},
    package_data=package_data,
)
