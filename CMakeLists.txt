cmake_minimum_required(VERSION 3.21)

project(gllm_extensions LANGUAGES CXX)

# CUDA by default, can be overridden by using -DGLLM_TARGET_DEVICE=... (used by setup.py)
set(GLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for gLLM")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${GLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")

# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")


# Supported/expected torch versions for CUDA.
set(TORCH_SUPPORTED_VERSION_CUDA "2.5.1")

#
# Try to find python package with an executable that exactly matches
# `GLLM_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (GLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${GLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set GLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# Ensure the 'nvcc' command is in the PATH
find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

#
# Forward the non-CUDA device extensions to external CMake scripts.
#
if (NOT GLLM_TARGET_DEVICE STREQUAL "cuda")
    message(FATAL_ERROR "Unsupported gLLM target device: ${GLLM_TARGET_DEVICE}")
    return()
endif()

#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if (CUDA_FOUND)
  set(VLLM_GPU_LANG "CUDA")

  if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
      "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
  clear_cuda_arches(CUDA_ARCH_FLAGS)
  extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
  message(STATUS "CUDA target architectures: ${CUDA_ARCHS}")
  # Filter the target architectures by the supported supported archs
  # since for some files we will build for all CUDA_ARCHS.
  cuda_archs_loose_intersection(CUDA_ARCHS
    "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
  message(STATUS "CUDA supported target architectures: ${CUDA_ARCHS}")
else()
  message(FATAL_ERROR "Can't find CUDA installation.")
endif()


#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})

#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

#
# Define extension targets
#

#
# _C extension
#

set(VLLM_EXT_SRC
  "csrc/cache_kernels.cu"
  "csrc/pos_encoding_kernels.cu"
  "csrc/activation_kernels.cu"
  "csrc/layernorm_kernels.cu"
  "csrc/torch_bindings.cpp")


define_gpu_extension_target(
  _C
  DESTINATION gllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR};${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)



# vLLM flash attention requires VLLM_GPU_ARCHES to contain the set of target
# arches in the CMake syntax (75-real, 89-virtual, etc), since we clear the
# arches in the CUDA case (and instead set the gencodes on a per file basis)
# we need to manually set VLLM_GPU_ARCHES here.
if(VLLM_GPU_LANG STREQUAL "CUDA")
  foreach(_ARCH ${CUDA_ARCHS})
    string(REPLACE "." "" _ARCH "${_ARCH}")
    list(APPEND VLLM_GPU_ARCHES "${_ARCH}-real")
  endforeach()
endif()

include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

#
# Build vLLM flash attention from source
#
# IMPORTANT: This has to be the last thing we do, because vllm-flash-attn uses the same macros/functions as vLLM.
# Because functions all belong to the global scope, vllm-flash-attn's functions overwrite vLLMs.
# They should be identical but if they aren't, this is a massive footgun.
#
# The vllm-flash-attn install rules are nested under vllm to make sure the library gets installed in the correct place.
# To only install vllm-flash-attn, use --component vllm_flash_attn_c.
# If no component is specified, vllm-flash-attn is still installed.

# If VLLM_FLASH_ATTN_SRC_DIR is set, vllm-flash-attn is installed from that directory instead of downloading.
# This is to enable local development of vllm-flash-attn within vLLM.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{VLLM_FLASH_ATTN_SRC_DIR})
  set(VLLM_FLASH_ATTN_SRC_DIR $ENV{VLLM_FLASH_ATTN_SRC_DIR})
endif()

if(VLLM_FLASH_ATTN_SRC_DIR)
  FetchContent_Declare(vllm-flash-attn SOURCE_DIR ${VLLM_FLASH_ATTN_SRC_DIR})
else()
  FetchContent_Declare(
          vllm-flash-attn
          GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
          GIT_TAG 96266b1111111f3d11aabefaf3bacbab6a89d03c
          GIT_PROGRESS TRUE
          # Don't share the vllm-flash-attn build between build types
          BINARY_DIR ${CMAKE_BINARY_DIR}/vllm-flash-attn
  )
endif()

# Set the parent build flag so that the vllm-flash-attn library does not redo compile flag and arch initialization.
set(VLLM_PARENT_BUILD ON)

# Ensure the gllm/vllm_flash_attn directory exists before installation
install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/gllm/vllm_flash_attn\")" COMPONENT vllm_flash_attn_c)

# Make sure vllm-flash-attn install rules are nested under gllm/
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY FALSE)" COMPONENT vllm_flash_attn_c)
install(CODE "set(OLD_CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}\")" COMPONENT vllm_flash_attn_c)
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}/gllm/\")" COMPONENT vllm_flash_attn_c)

# Fetch the vllm-flash-attn library
FetchContent_MakeAvailable(vllm-flash-attn)
message(STATUS "vllm-flash-attn is available at ${vllm-flash-attn_SOURCE_DIR}")

# Restore the install prefix
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${OLD_CMAKE_INSTALL_PREFIX}\")" COMPONENT vllm_flash_attn_c)
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" COMPONENT vllm_flash_attn_c)

# Copy over the vllm-flash-attn python files
install(
        DIRECTORY ${vllm-flash-attn_SOURCE_DIR}/vllm_flash_attn/
        DESTINATION gllm/vllm_flash_attn
        COMPONENT vllm_flash_attn_c
        FILES_MATCHING PATTERN "*.py"
)

# Nothing after vllm-flash-attn, see comment about macros above
