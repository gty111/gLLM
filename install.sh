#!/bin/bash

# Install the specified version of torch
pip install torch==2.8.0

# Download vLLM wheel
WHEEL_URL="https://wheels.vllm.ai/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm-0.11.0%2Bcu129-cp38-abi3-manylinux1_x86_64.whl"
WHEEL_FILE="vllm-0.11.0+cu129-cp38-abi3-manylinux1_x86_64.whl"

wget "$WHEEL_URL" -O "$WHEEL_FILE"
# Check if wget succeeded
if [ $? -ne 0 ]; then
    echo "Error: Failed to download $WHEEL_FILE. Please check your network or URL."
    exit 1
fi

# Set environment variable
export GLLM_PRECOMPILED_WHEEL_LOCATION=$(readlink -f "$WHEEL_FILE")

# Install the package in the current directory
pip install -v -e .
