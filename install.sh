#!/bin/bash

# Download vLLM wheel
WHEEL_URL="https://wheels.vllm.ai/0.11.0/vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl"
WHEEL_FILE="vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl"

# Only download if file doesn't exist
if [ ! -f "$WHEEL_FILE" ]; then
    if ! wget "$WHEEL_URL" -O "$WHEEL_FILE"; then
        echo "Error: Failed to download $WHEEL_FILE. Please check your network or URL."
        exit 1
    fi
fi

# Set environment variable
export GLLM_PRECOMPILED_WHEEL_LOCATION=$(readlink -f "$WHEEL_FILE")

# Install the package in the current directory
pip install -v -e .

pip install psutil
pip install --no-cache-dir --no-build-isolation flash_attn