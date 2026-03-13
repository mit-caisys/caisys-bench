#!/bin/bash

MODEL="google/gemma-3-27b-it"
GPUS='"device=0,1"'
CPUS=""
TP_SIZE=2
RUNS_PER_CONFIG=1

EXPERIMENT_SCRIPT="./video-qa/run.sh"
VLLM_START_SCRIPT="./start_instance/start_vllm_test.sh"

chmod +x "$EXPERIMENT_SCRIPT"
chmod +x "$VLLM_START_SCRIPT"

if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN is not set in this shell. The start script might fail."
fi
export HF_TOKEN

CACHE_VALUES=(0 5) 

for UTIL in 0.5; do
    for MM_CACHE in "${CACHE_VALUES[@]}"; do
    
        for (( i=1; i<=RUNS_PER_CONFIG; i++ )); do
        
            echo "========================================================"
            echo "CONFIGURATION: GPU UTIL = $UTIL | MM CACHE = ${MM_CACHE}GB"
            echo "RUN: $i / $RUNS_PER_CONFIG"
            echo "========================================================"

            echo "Cleaning up old containers..."
            sudo docker rm -f vllm_experiment_container 2>/dev/null || true
            

            while [ -n "$(sudo docker ps -aq -f name=^/vllm_experiment_container$)" ]; do
                echo "Container name still locked. Retrying removal..."
                sudo docker rm -f vllm_experiment_container 2>/dev/null
                sleep 2
            done
            
            sleep 2

            echo "Launching vLLM (Run $i)..."

            "$VLLM_START_SCRIPT" "$GPUS" "$MODEL" "$TP_SIZE" "$CPUS" "$UTIL" "$MM_CACHE"
            
            if [ $? -ne 0 ]; then
                echo "CRITICAL ERROR: Start script returned a non-zero exit code."
                exit 1
            fi

            echo "Waiting for vLLM server..."
            attempt_counter=0
            max_attempts=60
            
            until curl --output /dev/null --silent --fail http://localhost:8000/health; do
                if [ ${attempt_counter} -eq ${max_attempts} ];then
                  echo "Max attempts reached. vLLM failed to start."
                  sudo docker logs vllm_experiment_container
                  sudo docker rm -f vllm_experiment_container
                  exit 1
                fi
                
                if ! sudo docker ps | grep -q vllm_experiment_container; then
                     echo "Docker container died unexpectedly!"
                     sudo docker logs vllm_experiment_container
                     exit 1
                fi

                printf '.'
                attempt_counter=$(($attempt_counter+1))
                sleep 10 
            done
            echo -e "\nvLLM Server is UP!"

            echo "Running experiment script..."
            "$EXPERIMENT_SCRIPT"

            echo "Run $i finished. removing vLLM container..."
            sudo docker rm -f vllm_experiment_container
            
            sleep 5
        done
    done
done

echo "All configurations and runs completed successfully."