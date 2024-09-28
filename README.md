<p align="center">
    <img src=doc/pic/gLLM.svg height=240>
</p>

<h3 align="center">
Lightweight, easy, fast and cheap LLM serving
</h3>


---

## What is gLLM?

Integreted with features like **continuous batching**, **paged attention**, **prefix caching** and **pipeline schedule**, gLLM provides basic functionality (offline/online inference and interactive chat) to support large language model inference. Adopting part of codebase from vLLM, gLLM provides **faster** offline/online inference speed than vLLM with **lightweight** overhead and **minimal** code base. You can also see gLLM as a LLM inference playground for doing experiment or academic research.

### Offline performance (Test on llama3-8b)

<img src=doc/pic/offline_throughput.svg width=500> 

<img src=doc/pic/latency_breakdown.svg width=500> 


### Online performance (Test on llama3.1-8b)

<img src=doc/pic/online_avg_latency.svg height=240>


### Install gLLM
```
pip install --verbose -e .
```

### Chat mode
```
python examples/chat.py --model-path $MODEL_PATH
```

### Offline batch inference
```
python examples/batch_inference.py --model-path $MODEL \
    --share-gpt-path $SHARE_GPT_PATH --num-prompt $NUM_PROMPT \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION
```

### Offline benchmark with gllm/vllm
```
# replace $BACKEND with gllm or vllm
python benchmarks/benchmark_throughput.py --model $MODEL \
    --dataset $DATASET --num-prompt $NUM_PROMPT --backend $BACKEND \
    --trust-remote-code --gpu-memory-utilization $GPU_MEMORY_UTILIZATION
```

### Launch online serving

```
# To enable pipeline schedule, add "--pipe-schedule"
# To enable prefix caching, add "--enable-prefix-caching"
python -m gllm.entrypoints.api_server --port $PORT --model-path $MODEL_PATH
```

### Client Completions
```
python examples/client.py
```

### Client Chat Completions
```
python examples/chat_client.py
```

### Online benchmark with gllm or vllm
```
python benchmarks/benchmark_serving.py --backend $BACKEND --model $MODEL \
        --dataset-name $DATASET_NAME --dataset-path $DATASET_PATH \
        --num-prompts $NUM_PROMPTS --port $PORT --trust-remote-code \
        --request-rate $REQUEST_RATE
```

### Online prefix benchmark with gllm or vllm
```
python benchmarks/benchmark_prefix_serving.py \
        --trust-remote-code --backend $BACKEND --dataset $DATASET \
        --model $MODEL --num-max-users $NUM_USERS \
        --num-min-rounds $NUM_MIN_ROUNDS \
        --num-max-rounds $NUM_MAX_ROUNDS \
        --seed $NUM_USERS --port $PORT \
        --input-len-min $INPUT_LEN_MIN --input-len-max $INPUT_LEN_MAX
```

## Pipeline Schedule

### Comparison between baseline schedule and pipeline schedule
<img src=doc/pic/pipeline_execution.svg height=240>

### Architecture of pipeline schedule
<img src=doc/pic/pipeline_architecture.svg height=240>

## Supported Models
> Note that gLLM only support loading model of .safetensor or .bin format from local disk

- Llama: llama2-7b, llama3-8b, llama3.1-8b and deepseek-coder
- ChatGLM: chatglm3-6b and glm4-9b
- Qwen2: qwen2-7b

## Limited functionality

- Do NOT support TP or PP
- Limited number of supported models
