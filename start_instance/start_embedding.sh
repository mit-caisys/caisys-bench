#!/bin/bash

# ==============================================================================
# VLLM Model Runner Script
#
# Description:
# This script launches a vLLM OpenAI-compatible server in a Docker container.
# It is designed to be flexible, allowing you to specify the GPU devices,
# Hugging Face model, and tensor-parallel size via command-line arguments.
#
# It also automatically adjusts container image versions and model-specific
# parameters for known models like Mistral and Gemma.
#
# Pre-requisites:
# 1. Docker and NVIDIA Docker Runtime must be installed.
# 2. You must have an environment variable `HF_TOKEN` set with your
#    Hugging Face read token.
#
# ==============================================================================
usage() {
    echo "Usage: $0 <gpu_devices> <model_name> <tensor_parallel_size> <cpus>"
    echo
    echo "Arguments:"
    echo "  gpu_devices           - The GPU devices to use, in Docker's '--gpus' format."
    echo "                          Example: '\"device=0\"' or '\"device=0,1\"'"
    echo "  model_name            - The Hugging Face model identifier."
    echo "                          Example: 'google/embeddinggemma-300m'"
    echo "  tensor_parallel_size  - The number of GPUs to use for tensor parallelism."
    echo "                          Example: 1"
    echo "  cpus                  - The CPU cores to use, in Docker's '--cpuset-cpus' format."
    echo "                          Example: '0-15' or '4,10,35'"
    echo
    echo "Example for EmbeddingGemma (2 GPU, CPU cores 0,1,2):"
    echo "  ./vllm_start.sh '\"device=0,1\"' 'google/embeddinggemma-300m' 2 '0-2'"
    exit 1
}

echo "$HF_TOKEN"

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Error: Incorrect number of arguments."
    usage
fi

GPUS=$1
MODEL=$2
TENSOR_PARALLEL_SIZE=$3
CPUS=$4

if [ "$HF_TOKEN" = "" ]; then
    echo "Error: The HF_TOKEN environment variable is not set."
    echo "Please set it before running the script: export HF_TOKEN='your_token_here'"
    exit 1
fi

echo "✓ Arguments and environment validated."

export HF_HOME="$HOME/.cache/huggingface"
echo "• Using Hugging Face cache directory: $HF_HOME"

DOCKER_OPTIONS=()
VLLM_ARGS=()

DOCKER_OPTIONS+=(
    --rm
    --cpuset-cpus="$CPUS"
    --runtime nvidia
    --gpus "$GPUS"
    --cap-add SYS_ADMIN
    -v "$HF_HOME:/root/.cache/huggingface"
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"
    --env "VLLM_ENABLE_V1_MULTIPROCESSING=0"
    -p 8080:8000
    --ipc=host
)

VLLM_ARGS+=(
    "$MODEL"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-num-seqs 1024
    --kv-cache-metrics
)

echo "• Configuring for model: $MODEL"
if [[ "$MODEL" == *"google/embeddinggemma"* ]]; then
    VLLM_ARGS+=(
        --gpu-memory-utilization 0.01
        --dtype bfloat16
        --hf_overrides \'{\"matryoshka_dimensions\":[128\,256\,512\,768]}\'
    )
    echo "  ✓ Detected EmbeddingGemma model. Using EmbeddingGemma-specific settings."
else
    echo "  ! Warning: Unrecognized model. Using default image '$VLLM_IMAGE'."
    echo "  ! You may need to add model-specific arguments to this script."
fi

VLLM_IMAGE="vllm/vllm-openai:v0.12.0-x86_64"

echo "• Assembling final command..."

FULL_COMMAND="sudo docker run ${DOCKER_OPTIONS[*]} $VLLM_IMAGE ${VLLM_ARGS[*]}"

echo "----------------------------------------------------------------------"
echo "Executing the following command:"
echo "$FULL_COMMAND"
echo "----------------------------------------------------------------------"

eval "$FULL_COMMAND"
