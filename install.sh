#!/bin/bash
set -e

# Install gllm in editable mode (pulls sgl-kernel, flashinfer, triton automatically)
pip install -e .
