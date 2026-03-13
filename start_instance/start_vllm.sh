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
    echo "                          Example: 'mistralai/Mistral-Small-3.2-24B-Instruct-2506'"
    echo "  tensor_parallel_size  - The number of GPUs to use for tensor parallelism."
    echo "                          Example: 1"
    echo "  cpus                  - The CPU cores to use, in Docker's '--cpuset-cpus' format."
    echo "                          Example: '0-15' or '4,10,35'"
    echo
    echo "Example for Mistral (2 GPU, CPU cores 0,1,2):"
    echo "  ./vllm_start.sh '\"device=0,1\"' 'mistralai/Mistral-Small-3.2-24B-Instruct-2506' 2 '0-2'"
    echo
    echo "Example for Gemma (1 GPU, CPU cores 3,7,10):"
    echo "  ./vllm_start.sh '\"device=1\"' 'google/gemma-3-27b-it' 1 '3,7,10'"
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
    -p 8000:8000
    --ipc=host
)

VLLM_ARGS+=(
    "$MODEL"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-num-seqs 1024
    --enable-auto-tool-choice
    --enable-prefix-caching
)

echo "• Configuring for model: $MODEL"
if [[ "$MODEL" == *"mistralai/Mistral-Small-3.2-24B-Instruct-2506"* ]]; then
    VLLM_ARGS+=(
        --tokenizer-mode "mistral"
        --config_format mistral
        --load_format mistral
        --tool-call-parser mistral
        # --limit_mm_per_prompt 'image=100'
    )
    echo "  ✓ Detected Mistral model. Using Mistral-specific settings."
elif [[ "$MODEL" == *"google/gemma-3-27b-it"* ]]; then
    VLLM_ARGS+=(
        --tool-call-parser "pythonic"
        --max-model-len 65536
    )
    echo "  ✓ Detected Gemma model. Using Gemma-specific settings."
elif [[ "$MODEL" == *"Qwen/Qwen3-Omni-30B-A3B-Thinking"* ]]; then
    VLLM_ARGS+=(
        --tool-call-parser qwen3
    )
    echo "  ✓ Detected Qwen model. Using Qwen-specific settings."
elif [[ "$MODEL" == *"meta-llama/Llama-3.1-8B-Instruct"* ]]; then
    VLLM_ARGS+=(
        --tool-call-parser llama3_json
        --chat-template examples/tool_chat_template_llama3.1_json.jinja
    )
    echo "  ✓ Detected Llama model. Using Llama-specific settings."
else
    echo "  ! Warning: Unrecognized model. Using default image '$VLLM_IMAGE'."
    echo "  ! You may need to add model-specific arguments to this script."
fi

# VLLM_IMAGE="vllm/vllm-openai:v0.9.1" # Default to a recent version
VLLM_IMAGE="vllm/vllm-openai:v0.12.0"

echo "• Assembling final command..."

FULL_COMMAND="sudo docker run ${DOCKER_OPTIONS[*]} $VLLM_IMAGE ${VLLM_ARGS[*]}"

echo "----------------------------------------------------------------------"
echo "Executing the following command:"
echo "$FULL_COMMAND"
echo "----------------------------------------------------------------------"

eval "$FULL_COMMAND"
