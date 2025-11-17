pip install torch==2.8.0
wget https://wheels.vllm.ai/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm-0.11.0%2Bcu129-cp38-abi3-manylinux1_x86_64.whl
export GLLM_PRECOMPILED_WHEEL_LOCATION=$(readlink -f vllm-0.11.0+cu129-cp38-abi3-manylinux1_x86_64.whl)
pip install -v -e .