import importlib.util
import io
import logging
import os
import re
import subprocess
import sys
import warnings
from shutil import which
from typing import Dict, List

import torch
from packaging.version import Version, parse
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import CUDA_HOME


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

GLLM_TARGET_DEVICE = os.getenv("GLLM_TARGET_DEVICE", "cuda")

# gLLM only supports Linux platform
assert sys.platform.startswith(
    "linux"), "gLLM only supports Linux platform (including WSL)."

MAIN_CUDA_VERSION = "12.1"


def is_sccache_available() -> bool:
    return which("sccache") is not None


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


class CMakeExtension(Extension):

    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=True, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: Dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = os.getenv("MAX_JOBS", None)
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()

        nvcc_threads = None
        if _is_cuda() and get_nvcc_cuda_version() >= Version("11.2"):
            # `nvcc_threads` is either the value of the NVCC_THREADS
            # environment variable (if defined) or 1.
            # when it is set, we reduce `num_jobs` to avoid
            # overloading the system.
            nvcc_threads = os.getenv("NVCC_THREADS", None)
            if nvcc_threads is not None:
                nvcc_threads = int(nvcc_threads)
                logger.info(
                    "Using NVCC_THREADS=%d as the number of nvcc threads.",
                    nvcc_threads)
            else:
                nvcc_threads = 1
            num_jobs = max(1, num_jobs // nvcc_threads)

        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = os.getenv("CMAKE_BUILD_TYPE") or default_cfg

        # where .so files will be written, should be the same for all extensions
        # that use the same CMakeLists.txt.
        outdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(outdir),
            '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}'.format(self.build_temp),
            '-DGLLM_TARGET_DEVICE={}'.format(GLLM_TARGET_DEVICE),
        ]

        verbose = bool(int(os.getenv('VERBOSE', '0')))
        if verbose:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=ON']

        if is_sccache_available():
            cmake_args += [
                '-DCMAKE_CXX_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_C_COMPILER_LAUNCHER=sccache',
            ]
        elif is_ccache_available():
            cmake_args += [
                '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache',
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ['-DGLLM_PYTHON_EXECUTABLE={}'.format(sys.executable)]

        #
        # Setup parallelism and build tool
        #
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ['-DNVCC_THREADS={}'.format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ['-G', 'Ninja']
            cmake_args += [
                '-DCMAKE_JOB_POOL_COMPILE:STRING=compile',
                '-DCMAKE_JOB_POOLS:STRING=compile={}'.format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []
        subprocess.check_call(
            ['cmake', ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp)

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError as e:
            raise RuntimeError('Cannot find CMake executable') from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []
        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(remove_prefix(ext.name, "gllm."))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return GLLM_TARGET_DEVICE == "cuda" and has_cuda

def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_gllm_version() -> str:
    version = '0.0.1'

    cuda_version = str(get_nvcc_cuda_version())
    if cuda_version != MAIN_CUDA_VERSION:
        cuda_version_str = cuda_version.replace(".", "")[:3]
        version += f"+cu{cuda_version_str}"

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
    cuda_major, cuda_minor = torch.version.cuda.split(".")
    modified_requirements = []
    for req in requirements:
        if ("vllm-flash-attn" in req
                and not (cuda_major == "12" and cuda_minor == "1")):
            # vllm-flash-attn is built only for CUDA 12.1.
            # Skip for other versions.
            continue
        modified_requirements.append(req)
    requirements = modified_requirements
    
    return requirements


ext_modules = []

ext_modules.append(CMakeExtension(name="gllm._C"))


setup(
    name="gllm",
    version=get_gllm_version(),
    author="gtyinstinct",
    license="Apache 2.0",
    description=("A high-throughput and memory-efficient inference and "
                 "serving engine for LLMs"),
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
    packages=find_packages(exclude=("benchmarks", "csrc", "docs", "examples",
                                    "tests*")),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": cmake_build_ext},
)
