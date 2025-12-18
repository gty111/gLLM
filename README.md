<p align="center">
    <img src=doc/pic/gLLM.svg height=600>
</p>

<h4 align="center">
Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling
</h4>


---

## What is gLLM?

<p align="center">
<img src=doc/pic/overview.svg width=500>
</p>

Integreted with features like **continuous batching**, **paged attention**, **chunked prefill**, **prefix caching**, **cuda graph**, **token throttling**, **pipeline parallelism**, **expert parallelsim** and **tensor parallelism**, gLLM provides basic functionality (**offline/online inference and interactive chat**) to deploy distributed LLMs (**supported in huggingface**) inference. gLLM provides **equivalent or superior** offline/online inference speed with mainstream inference engine and **minimal** code base. You can also see gLLM as a LLM inference playground for doing experiment or academic research.

*Latest News* :fire:
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

<details>
<summary>Previous News</summary>

- [2025/04/27]: We support multi-node deployments. You can serve your model across different machines :blush:
- [2025/04/21]: We release our paper on [arXiv:2504.14775](https://arxiv.org/abs/2504.14775) :partying_face:
- [2025/03/15]: Chunked prefill has been integrated. You can input any length of text you want :hugs:
- [2025/03/01]: Pipeline parallelism has been integrated. You can run any size of model you want :laughing:
- [2025/02/27]: We apply numerous optimizations which lowers CPU overhead a lot :clap:

</details>

## Token Throttling

### Prefill Token Throttling
<p align="center">
<img src=doc/pic/prefill_throttling.svg >
</p>

---

### Decode Token Throttling
<p align="center">
<img src=doc/pic/decode_throttling.svg >
</p>

## Install gLLM
```
bash install.sh
```

## Quickstart

### Interactive Offline Chat
```
python examples/chat.py --model $MODEL_PATH
```

### Offline Batch Inference
```
python examples/batch_inference.py --model $MODEL \
    --share-gpt-path $SHARE_GPT_PATH --num-prompt $NUM_PROMPT \
    --gpu-memory-util $GPU_MEMORY_UTIL
```

### Offline Benchmark
```
python benchmarks/benchmark_throughput.py --model $MODEL \
    --dataset $SHAREGPT_PATH --num-prompt $NUM_PROMPT --backend gllm \
    --gpu-memory-util $GPU_MEMORY_UTIL
```

### Launch OpenAI-Compatible Server (Intra-node)

```
# To see the description of args, run 'python -m gllm.entrypoints.api_server -h'
python -m gllm.entrypoints.api_server --port $PORT --model-path $MODEL_PATH \
    --enable-prefix-caching --pp $PP --tp $TP
```

### Launch OpenAI-Compatible Server (Multi-node)

gLLM can be launched in three modes: (1) `normal`, used for single-node multiple GPUs (2) `master`, used for multi-node deployment (3) `slave`, used for multi-node deployment.

To launch master gLLM instance
```
python -m gllm.entrypoints.api_server --port $PORT --master-port $MASTER_PORT \
    --model-path $MODEL_PATH --pp $PP --launch-mode master --worker-ranks $RANKS
```
To launch slave gLLM instance
```
python -m gllm.entrypoints.api_server --host $HOST \
    --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
    --model-path $MODEL_PATH --pp $PP --launch-mode slave --worker-ranks $RANKS
```
There are something you need to care about
- Make sure `$MASTER_PORT` and `$MASTER_ADDR` in slave instance can be matched to that in master instance
- Make sure slave instance can set up connection with master instance using `$MASTER_ADDR`
- Make sure master instance can set up connection with slave instance using `$HOST`
- Make sure `$PP` can be matched to `$RANKS` in slave or master instance
    - For example, we want to launch two gLLM instances, `$PP` is set to `4`, `$RANKS` in master is set to `0,1`, then `$RANKS` in slave must set to `2,3`
- Make sure set environment variable `NCCL_SOCKET_IFNAME` `NCCL_IB_DISABLE` properly

### Client Completions
```
# Launch server first
python examples/client.py --port $PORT
```

### Interactive Online Chat
```
# Launch server first
python examples/chat_client.py --port $PORT
```

### Online Benchmark
```
# Launch server first
python benchmarks/benchmark_serving.py --backend $BACKEND --model $MODEL \
        --dataset-name $DATASET_NAME --dataset-path $DATASET_PATH \
        --num-prompts $NUM_PROMPTS --port $PORT --trust-remote-code \
        --request-rate $REQUEST_RATE
```

### Online Prefix Benchmark
```
# Launch server first
python benchmarks/benchmark_prefix_serving.py \
        --trust-remote-code --backend $BACKEND --dataset $SHAREGPT_PATH \
        --model $MODEL --num-max-users $NUM_USERS \
        --num-min-rounds $NUM_MIN_ROUNDS \
        --num-max-rounds $NUM_MAX_ROUNDS \
        --port $PORT
```

### Evaluate Output Quality
```
# Launch server first
python benchmarks/evaluate_MMLU_pro.py --model $MODEL
```

## Supported Models

- DeepSeek Series: DeepSeek R1, DeepSeek V3, DeepSeek V2
- Qwen Series: Qwen3, Qwen2.5 VL, Qwen2.5, Qwen2
- Llama Series: Llama3.2, Llama3.1, Llama3, Llama2 and deepseek-coder
- Mixtral Series: Mixtral-8x7B, Mixtral-8x22B
- ChatGLM Series: Glm4 and Chatglm3

## Supported Quantization Method

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

We studied the architecture and implemented code reuse from these existing projects: [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang) and [TD-Pipe]().
