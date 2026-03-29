#!/bin/bash

if [ "$#" -lt 4 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage: $0 <GPUS> <MODEL> <TENSOR_PARALLEL_SIZE> <CPUS> [GPU_MEM_UTIL] [MM_CACHE_GB]"
    exit 1
fi

GPUS=$1
MODEL=$2
TENSOR_PARALLEL_SIZE=$3
CPUS=$4

GPU_MEM_UTIL=${5:-}
MM_CACHE_GB=${6:-}

if [ "$HF_TOKEN" = "" ]; then
    echo "Error: The HF_TOKEN environment variable is not set."
    exit 1
fi

export HF_HOME="$HOME/.cache/huggingface"

DOCKER_OPTIONS=()
DOCKER_OPTIONS+=(
    --rm
    -d
    --name vllm_experiment_container
    --runtime nvidia
    --gpus "$GPUS"          
    --cap-add SYS_ADMIN
    -v "$HF_HOME:/root/.cache/huggingface"
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"
    --env "VLLM_ENABLE_V1_MULTIPROCESSING=0"
    -p 8000:8000
    --ipc=host
    --env "TORCH_COMPILE_DISABLE=1"
    --env "VLLM_GUIDED_DECODING_BACKEND=xgrammar"
)

if [ -n "$CPUS" ] && [ "$CPUS" != "all" ]; then
    echo "• Setting CPU affinity to: $CPUS"
    DOCKER_OPTIONS+=( --cpuset-cpus="$CPUS" )
else
    echo "• No specific CPU affinity set. Using all available CPUs."
fi

VLLM_ARGS=()
VLLM_ARGS+=(
    "$MODEL"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-num-seqs 1024
    --enable-auto-tool-choice
    --enable-prefix-caching
    --enforce-eager
    --trust-remote-code
)

if [ -n "$GPU_MEM_UTIL" ]; then
    echo "• Setting GPU Memory Utilization to: $GPU_MEM_UTIL"
    VLLM_ARGS+=( --gpu-memory-utilization "$GPU_MEM_UTIL" )
fi

if [ -n "$MM_CACHE_GB" ]; then
    echo "• Setting MM Processor Cache to: ${MM_CACHE_GB}GB"
    VLLM_ARGS+=( 
        --mm-processor-cache-gb "$MM_CACHE_GB"
        --mm-processor-cache-type shm 
    )
fi

echo "• Configuring for model: $MODEL"
if [[ "$MODEL" == *"mistralai/Mistral-Small-3.2-24B-Instruct-2506"* ]]; then
    VLLM_ARGS+=( --tokenizer-mode "mistral" --config_format mistral --load_format mistral --tool-call-parser mistral )
elif [[ "$MODEL" == *"google/gemma-3-27b-it"* ]]; then
    VLLM_ARGS+=( --tool-call-parser "pythonic" --max-model-len 50000)
elif [[ "$MODEL" == *"Qwen/Qwen3-Omni-30B-A3B-Thinking"* ]]; then
    VLLM_ARGS+=( --tool-call-parser qwen3 )
elif [[ "$MODEL" == *"meta-llama/Llama-3.1-8B-Instruct"* ]]; then
    VLLM_ARGS+=( --tool-call-parser llama3_json --chat-template examples/tool_chat_template_llama3.1_json.jinja )
fi

VLLM_IMAGE="vllm/vllm-openai:v0.18.0"

echo "----------------------------------------------------------------------"
echo "Starting vLLM Container..."
echo "----------------------------------------------------------------------"

if [ -e /dev/shm/VLLM_OBJECT_STORAGE_SHM_BUFFER ]; then
    sudo rm -f /dev/shm/VLLM_OBJECT_STORAGE_SHM_BUFFER
    echo "• Removed stale SHM buffer."
fi

sudo docker run \
    "${DOCKER_OPTIONS[@]}" \
    "$VLLM_IMAGE" \
    "${VLLM_ARGS[@]}"