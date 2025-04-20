<p align="center">
    <img src=doc/pic/gLLM.svg height=240>
</p>

<h4 align="center">
Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling
</h4>


---

## What is gLLM?

Integreted with features like **continuous batching**, **paged attention**, **chunked prefill**, **prefix caching** and **pipeline parallelism**, gLLM provides basic functionality (offline/online inference and interactive chat) to support large language model inference. gLLM provides **equivalent or superior** offline/online inference speed with mainstream inference engine and **minimal** code base. You can also see gLLM as a LLM inference playground for doing experiment or academic research.

*Latest News* :fire:
- [2025/03/15]: Chunked prefill has been integrated. You can input any length of text you want :hugs:
- [2025/03/01]: Pipeline parallelism has been integrated. You can run any size of model you want :laughing: 
- [2025/02/27]: We apply numerous optimizations which lowers CPU overhead a lot :clap: 


## Install gLLM
```
pip install torch==2.5.1
pip install -v -e .
```

## Quickstart

### Interactive Offline Chat
```
python examples/chat.py --model-path $MODEL_PATH
```

### Offline Batch Inference
```
python examples/batch_inference.py --model-path $MODEL \
    --share-gpt-path $SHARE_GPT_PATH --num-prompt $NUM_PROMPT \
    --gpu-memory-util $GPU_MEMORY_UTIL
```

### Offline Benchmark
```
python benchmarks/benchmark_throughput.py --model $MODEL \
    --dataset $SHAREGPT_PATH --num-prompt $NUM_PROMPT --backend gllm \
    --gpu-memory-util $GPU_MEMORY_UTIL
```

### Launch Online Server

```
# To see the description of args, run 'python -m gllm.entrypoints.api_server -h'
python -m gllm.entrypoints.api_server --port $PORT --model-path $MODEL_PATH \
    --enable-prefix-caching --pp $PP_STAGES
```

### Client Completions
```
# Launch online server first
python examples/client.py --port $PORT
```

### Interactive Online Chat
```
# Launch online server first
python examples/chat_client.py --port $PORT
```

### Online Benchmark
```
# Launch online server first
python benchmarks/benchmark_serving.py --backend $BACKEND --model $MODEL \
        --dataset-name $DATASET_NAME --dataset-path $DATASET_PATH \
        --num-prompts $NUM_PROMPTS --port $PORT --trust-remote-code \
        --request-rate $REQUEST_RATE
```

### Online Prefix Benchmark
```
# Launch online server first
python benchmarks/benchmark_prefix_serving.py \
        --trust-remote-code --backend $BACKEND --dataset $SHAREGPT_PATH \
        --model $MODEL --num-max-users $NUM_USERS \
        --num-min-rounds $NUM_MIN_ROUNDS \
        --num-max-rounds $NUM_MAX_ROUNDS \
        --port $PORT 
```

### Evaluate Output Quality
```
# Launch online server first
python evaluations/evaluate_MMLU_pro.py --model $MODEL --port $PORT
```

## Supported Models

- Llama Series: Llama2, Llama3, Llama3.1 and deepseek-coder
- ChatGLM Series: Chatglm3 and glm4
- Qwen2 Series: Qwen2, Qwen2.5

## Roadmap

- [ ] Support TP
- [ ] Support mult-node deployment
- [ ] Support as many models as possible
