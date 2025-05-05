import importlib.util
import io
import logging
import os
import subprocess
import sys
from pathlib import Path
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
        num_jobs = None
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = min(os.cpu_count(), 32)

        nvcc_threads = None
        if _is_cuda() and get_nvcc_cuda_version() >= Version("11.2"):
            # `nvcc_threads` is either the value of the NVCC_THREADS
            # environment variable (if defined) or 1.
            # when it is set, we reduce `num_jobs` to avoid
            # overloading the system.
            nvcc_threads = None
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

        cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),
        ]

        if is_sccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_HIP_COMPILER_LAUNCHER=sccache',
            ]
        elif is_ccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_HIP_COMPILER_LAUNCHER=ccache',
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ['-DGLLM_PYTHON_EXECUTABLE={}'.format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ['-DVLLM_PYTHON_PATH={}'.format(":".join(sys.path))]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ['-DFETCHCONTENT_BASE_DIR={}'.format(fc_base_dir)]

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

        def target_name(s: str) -> str:
            return s.removeprefix("gllm.").removeprefix("vllm_flash_attn.")

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == self.build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            prefix = outdir
            for i in range(ext.name.count('.')):
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = [
                "cmake", "--install", ".", "--prefix", prefix, "--component",
                target_name(ext.name)
            ]

            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self):
        # First, run the standard build_ext command to compile the extensions
        super().run()

        # copy vllm/vllm_flash_attn/*.py from self.build_lib to current
        # directory so that they can be included in the editable build
        import glob
        files = glob.glob(
            os.path.join(self.build_lib, "gllm", "vllm_flash_attn", "*.py"))
        for file in files:
            dst_file = os.path.join("gllm/vllm_flash_attn",
                                    os.path.basename(file))
            print(f"Copying {file} to {dst_file}")
            self.copy_file(file, dst_file)


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

    return requirements


ext_modules = []

ext_modules.append(CMakeExtension(name="gllm._C"))
ext_modules.append(CMakeExtension(name="gllm._moe_C"))
ext_modules.append(
    CMakeExtension(name="gllm.vllm_flash_attn.vllm_flash_attn_c"))


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
