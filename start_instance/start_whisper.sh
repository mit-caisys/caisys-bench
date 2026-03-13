#!/bin/bash
set -ex

export HF_HOME="$HOME/.cache/huggingface"

echo $HF_HOME

start_whisper() {
  hardware=$1
  VOLUME="${HF_HOME}:/root/.cache/huggingface"
  WHISPER__MODEL="Systran/faster-whisper-tiny.en"
  WHISPER__TTL=-1
  WHISPER__NUM_WORKERS=96
  WHISPER__USE_BATCHED_MODE="True"
  if [[ "$hardware" == "gpu" ]]; then
    gpu_list=$2
    export CUDA_VISIBLE_DEVICES=${gpu_list}
    sudo docker run --detach \
                    --name murakkab-whisper \
                    --gpus="all" \
                    --publish 7771:8000 \
                    --volume ${VOLUME} \
                    --env WHISPER__INFERENCE_DEVICE="cuda" \
                    --env WHISPER__DEVICE_INDEX="[${gpu_list}]" \
                    --env WHISPER__MODEL=${WHISPER__MODEL} \
                    --env WHISPER__TTL=${WHISPER__TTL} \
                    --env WHISPER__NUM_WORKERS=${WHISPER__NUM_WORKERS} \
                    --env WHISPER__USE_BATCHED_MODE=${WHISPER__USE_BATCHED_MODE} \
                    ghcr.io/speaches-ai/speaches:latest-cuda
  elif [[ "$hardware" == "cpu" ]]; then
    WHISPER__NUM_WORKERS=8
    WHISPER__CPU_THREADS=16
    sudo docker run --detach \
                    --name murakkab-whisper \
                    --publish 7771:8000 \
                    --volume ${VOLUME} \
                    --env WHISPER__INFERENCE_DEVICE="cpu" \
                    --env WHISPER__MODEL=${WHISPER__MODEL} \
                    --env WHISPER__TTL=${WHISPER__TTL} \
                    --env WHISPER__NUM_WORKERS=${WHISPER__NUM_WORKERS} \
                    --env WHISPER__CPU_THREADS=${WHISPER__CPU_THREADS} \
                    --env WHISPER__USE_BATCHED_MODE=${WHISPER__USE_BATCHED_MODE} \
                    ghcr.io/speaches-ai/speaches:latest-cpu
  else
    echo "Unsupported hardware type"
    exit 1
  fi
  sudo docker container ls
  echo "Whisper started..."
}

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  $0 cpu"
  echo "  $0 gpu <comma_separated_list_of_gpus>"
  exit 1
fi

mode=$1

if [ "$mode" = "cpu" ]; then
  if [ $# -ne 1 ]; then
    echo "Error: 'cpu' mode does not take additional arguments."
    exit 1
  fi
  echo "Running in CPU mode."
  start_whisper $mode

elif [ "$mode" = "gpu" ]; then
  if [ $# -ne 2 ]; then
    echo "Error: 'gpu' mode requires a comma-separated list of GPUs."
    echo "Usage: $0 gpu 0,1,2"
    exit 1
  fi
  gpu_list=$2
  echo "Running in GPU mode with GPUs: $gpu_list"
  start_whisper $mode $gpu_list

else
  echo "Error: First argument must be either 'cpu' or 'gpu'."
  exit 1
fi