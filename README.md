<p align="center">
    <img src=doc/pic/gLLM.svg height=240>
</p>

<h4 align="center">
Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling
</h4>


---

## What is gLLM?

<p align="center">
<img src=doc/pic/overview.svg width=500>
</p>

Integreted with features like **continuous batching**, **paged attention**, **chunked prefill**, **prefix caching**, **token throttling**, **pipeline parallelism**, **expert parallelsim** and **tensor parallelism**, gLLM provides basic functionality (**offline/online inference and interactive chat**) to deploy distributed LLMs (**supported in huggingface**) inference. gLLM provides **equivalent or superior** offline/online inference speed with mainstream inference engine and **minimal** (~6k loc) code base. You can also see gLLM as a LLM inference playground for doing experiment or academic research.

*Latest News* :fire:
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
pip install torch==2.5.1
pip install -v -e .
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

> Experimental feature

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
python evaluations/evaluate_MMLU_pro.py --model $MODEL --port $PORT
```

## Supported Models

- Qwen Series: Qwen3, Qwen2.5, Qwen2
- Llama Series: Llama3.2, Llama3.1, Llama3, Llama2 and deepseek-coder
- Mixtral Series: Mixtral-8x7B, Mixtral-8x22B
- ChatGLM Series: Glm4 and Chatglm3 

## Roadmap

- [ ] Support more models


## Cite Our Work
```
@misc{guo2025gllmglobalbalancedpipeline,
      title={gLLM: Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling}, 
      author={Tianyu Guo and Xianwei Zhang and Jiangsu Du and Zhiguang Chen and Nong Xiao and Yutong Lu},
      year={2025},
      eprint={2504.14775},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2504.14775}, 
}
```

## Acknowledgment

We studied the architecture and implemented code reuse from these existing projects: [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang) and [TD-Pipe]().