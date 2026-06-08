<p align="center">
    <img src=docs/pic/gLLM.svg height=600>
</p>

<h4 align="center">
An Efficient and Versatile Inference Engine for Distributed LLM Serving
</h4>


---

## What is gLLM?

gLLM is an efficient and versatile inference engine for distributed LLM serving. It supports a wide range of models (**dense**, **MoE**, **multimodal/vision-language**, and **hybrid-attention** architectures from HuggingFace) and deployment scenarios (**offline/online inference and interactive chat**). Under the hood, gLLM integrates features like **continuous batching**, **paged attention**, **chunked prefill**, **prefix caching**, **cuda graph**, **token throttling**, **pipeline parallelism**, **expert parallelism** and **tensor parallelism**, delivering **equivalent or superior** inference speed compared to mainstream inference engines while keeping a **minimal** code base. You can also see gLLM as an LLM inference playground for experiments or academic research.

*Latest News* :fire:
- [2026/06/05]: [Encoder disaggregation](docs/encoder_disaggregation_usage.md) is supported, decoupling the multimodal encoder from the LLM for better resource utilization :rocket:
- [2026/05/27]: Qwen3.5/3.6 is supported — dense, MoE, VL, and FP8 :tada:


<details>
<summary>Previous News</summary>

- [2025/12/04]: Cuda graph is supported :tada:
- [2025/11/19]: DeepSeek V3/R1 is supported :laughing:
- [2025/09/19]: [DynaPipe](https://openreview.net/forum?id=D6w7wIN360) is accepted by NeurIPS'25. Congratulations :smiling_face_with_three_hearts:
- [2025/08/15]: Qwen2.5 VL is supported :hugs:
- [2025/08/01]: DeepSeek V2 is supported :clap:
- [2025/07/12]: FP8 quantization for Qwen3/2.5 is supported :tada:
- [2025/06/27]: [gLLM](https://doi.org/10.1145/3712285.3759823) is accepted by SC'25. Congratulations :smiling_face_with_three_hearts:
- [2025/06/21]: Expert parallelism is integrated :heart_eyes:
- [2025/06/14]: Tensor parallelism is now integrated, allowing joint deploying with pipeline parallelism :sunglasses:
- [2025/05/05]: MoE architecture is supported. Try Qwen2/3 MoE models :star_struck:
- [2025/04/29]: Qwen3 day 1 support. Come and try Qwen3 :tada:
- [2025/04/27]: gLLM is open sourced :earth_asia:
- [2025/04/27]: We support multi-node deployments. You can serve your model across different machines :blush:
- [2025/04/21]: We release our paper on [arXiv:2504.14775](https://arxiv.org/abs/2504.14775) :partying_face:
- [2025/03/15]: Chunked prefill has been integrated. You can input any length of text you want :hugs:
- [2025/03/01]: Pipeline parallelism has been integrated. You can run any size of model you want :laughing:
- [2025/02/27]: We apply numerous optimizations which lowers CPU overhead a lot :clap:

</details>

## Key Features

- **Broad model support**: dense, MoE, multimodal/vision-language, and hybrid-attention architectures from HuggingFace, including FP8 checkpoints.
- **Flexible parallelism**: pipeline, tensor, and expert parallelism that can be freely combined for single- or multi-node deployments.
- **High-performance execution**: continuous batching, paged attention, chunked prefill, prefix caching, and cuda graph.
- **Balanced scheduling**: token throttling for smoother pipeline utilization across prefill and decode (see below).
- **Versatile serving**: offline batch inference, online serving, and interactive chat.
- **Minimal codebase**: equivalent or superior speed to mainstream engines, while staying easy to read, hack, and extend.

## Installation

- For development:
```bash
uv pip install -e .
```

- For release:
```bash
uv pip install "git+https://github.com/gty111/gLLM.git"
```

## Quickstart

### Interactive Offline Chat
```bash
python examples/chat.py --model $MODEL_PATH
```

### Offline Batch Inference
```bash
python examples/batch_inference.py --model $MODEL \
    --share-gpt-path $SHARE_GPT_PATH --num-prompt $NUM_PROMPT \
    --gpu-memory-util $GPU_MEMORY_UTIL
```

### Offline Benchmark
```bash
python benchmarks/benchmark_throughput.py --model $MODEL \
    --dataset $SHAREGPT_PATH --num-prompt $NUM_PROMPT --backend gllm \
    --gpu-memory-util $GPU_MEMORY_UTIL
```

### Launch OpenAI-Compatible Server (Intra-node)

```bash
# To see the description of args, run 'python -m gllm.entrypoints.api_server -h'
python -m gllm.entrypoints.api_server --port $PORT --model-path $MODEL_PATH \
    --enable-prefix-caching --pp $PP --tp $TP
```

### Launch OpenAI-Compatible Server (Multi-node)

gLLM supports three launch modes: (1) `normal` for single-node multiple GPUs, (2) `master` for multi-node deployment, and (3) `slave` for multi-node deployment.

To launch the master instance:
```bash
python -m gllm.entrypoints.api_server --port $PORT --master-port $MASTER_PORT \
    --model-path $MODEL_PATH --pp $PP --launch-mode master --worker-ranks $RANKS
```
To launch the slave instance:
```bash
python -m gllm.entrypoints.api_server --host $HOST \
    --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
    --model-path $MODEL_PATH --pp $PP --launch-mode slave --worker-ranks $RANKS
```

> **Notes:**
> - Ensure `$MASTER_PORT` and `$MASTER_ADDR` in the slave instance match those in the master instance
> - Ensure the slave instance can connect to the master instance using `$MASTER_ADDR`
> - Ensure the master instance can connect to the slave instance using `$HOST`
> - Ensure `$PP` matches `$RANKS` across instances (e.g., if `$PP=4` and master has `$RANKS=0,1`, then slave must have `$RANKS=2,3`)
> - Set environment variables `NCCL_SOCKET_IFNAME` and `NCCL_IB_DISABLE` properly

### Client Completions
```bash
# Launch server first
python examples/client.py --port $PORT
```

### Interactive Online Chat
```bash
# Launch server first
python examples/chat_client.py --port $PORT
```

### Evaluate Output Quality
```bash
# Launch server first
python benchmarks/evaluate_MMLU_pro.py --model $MODEL
```

## Supported Models

- Kimi Series: Moonlight, K2-Base, K2-Instruct
- DeepSeek Series: DeepSeek R1, DeepSeek V3, DeepSeek V2
- Qwen Series: Qwen3.6, Qwen3.5, Qwen3 VL, Qwen3, Qwen2.5 VL, Qwen2.5, Qwen2
- Llama Series: Llama3.2, Llama3.1, Llama3, Llama2 and deepseek-coder
- Mixtral Series: Mixtral-8x7B, Mixtral-8x22B
- ChatGLM Series: Glm4 and Chatglm3

## Supported Quantization Methods

- fp8

## Roadmap

- [ ] Support more models
- [ ] Support more quantization methods


## Cite Our Work

```
@inproceedings{10.1145/3712285.3759823,
author = {Guo, Tianyu and Zhang, Xianwei and Du, Jiangsu and Chen, Zhiguang and Xiao, Nong and Lu, Yutong},
title = {gLLM: Global Balanced Pipeline Parallelism Systems for Distributed LLMs Serving with Token Throttling},
year = {2025},
isbn = {9798400714665},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3712285.3759823},
doi = {10.1145/3712285.3759823},
booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
pages = {1725–1741},
numpages = {17},
keywords = {LLM, Inference, Parallelism, Pipeline Bubbles, Throttling, Runtime},
location = {},
series = {SC '25}
}
```

```
@inproceedings{
xu2025dynapipe,
title={DynaPipe: Dynamic Layer Redistribution for Efficient Serving of {LLM}s with Pipeline Parallelism},
author={HongXin Xu and Tianyu Guo and Xianwei Zhang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=D6w7wIN360}
}
```

## Acknowledgment

We studied the architecture and reused code from these existing projects: [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang) and [TD-Pipe]().
