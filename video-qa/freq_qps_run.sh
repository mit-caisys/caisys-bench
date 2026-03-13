#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DIR_PATH="$SCRIPT_DIR"
PARENT_DIR=$( dirname "$SCRIPT_DIR" )

VLLM_START_SCRIPT="$PARENT_DIR/start_instance/start_vllm_test.sh"
VLLM_GPU_ID=0
STT_GPU_ID=1

MODEL="google/gemma-3-27b-it"
GPUS='"device=0"'
CPUS=""
TP_SIZE=1

STT_START_SCRIPT="$PARENT_DIR/start_instance/start_whisper.sh"

FREQ_MIN=300
FREQ_LOW=570
FREQ_MID=855
FREQ_HIGH=1125
FREQ_MAX=1410

# --- NEW: Added NUM_TRIALS for experimental variance ---
if [[ "$1" == "--quick" ]]; then
    echo ">>> QUICK MODE ENABLED: 'min'/'max' frequencies, 1 Trial."
    FREQ_VALUES=($FREQ_MIN $FREQ_MAX)
    FREQ_LABELS=("min" "max")
    NUM_TRIALS=1
else
    echo ">>> FULL MODE ENABLED: All 5 frequencies, 3 Trials."
    FREQ_VALUES=($FREQ_MIN $FREQ_LOW $FREQ_MID $FREQ_HIGH $FREQ_MAX)
    FREQ_LABELS=("min" "low" "mid" "high" "max")
    NUM_TRIALS=3
fi

QPS_VALUES=(0.2 0.4 0.6)

MONITORING_FOLDER="$PARENT_DIR/monitoring"
PROCESS_FOLDER="$PARENT_DIR/process_data"
CONFIG_FILE="$DIR_PATH/config.json"

FILE_FORMAT="frames-*_transcription-*.csv"
LATENCY_NODES="encode_videos speech_to_text multimodal_model end_to_end end_to_end_load_gen"
VALUE_NODES="prompt_tokens total_tokens"
TIMELINE_NODES="encode_videos speech_to_text multimodal_model"

BATCH_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
MASTER_OUTPUT_DIR="$SCRIPT_DIR/freq_exp_results_$BATCH_TIMESTAMP"
mkdir -p "$MASTER_OUTPUT_DIR"

wait_for_service() {
    local url=$1
    local name=$2
    local max_retries=60 
    local count=0

    echo "Waiting for $name to be ready at $url..."
    until curl -s -f "$url" > /dev/null; do
        sleep 5
        count=$((count+1))
        echo "  ...waiting for $name ($count/$max_retries)..."
        if [ $count -ge $max_retries ]; then
            echo "Error: Timed out waiting for $name."
            exit 1
        fi
    done
    echo "$name is Ready!"
}

stop_services() {
    echo "------------------------------------------------"
    echo "Stopping Services..."
    echo "------------------------------------------------"
    if [ "$(sudo docker ps -q -f name=vllm_experiment_container)" ]; then
        sudo docker stop vllm_experiment_container
    fi
    sudo docker rm vllm_experiment_container 2>/dev/null || true

    if [ "$(sudo docker ps -q -f name=murakkab-whisper)" ]; then
        sudo docker stop murakkab-whisper
    fi
    sudo docker rm murakkab-whisper 2>/dev/null || true
    echo "All services stopped."
}

start_services() {
    echo "------------------------------------------------"
    echo "PHASE 0: Starting Services"
    echo "------------------------------------------------"
    echo "Launching vLLM on GPU $VLLM_GPU_ID..."
    bash "$VLLM_START_SCRIPT" "$GPUS" "$MODEL" "$TP_SIZE" "$CPUS"

    echo "Launching STT (Whisper) on GPU $STT_GPU_ID..."
    bash "$STT_START_SCRIPT" gpu "$STT_GPU_ID"

    wait_for_service "http://localhost:8000/health" "vLLM"
    echo "Waiting 10s for STT to stabilize..."
    sleep 10
    echo "Services started successfully."
}

set_gpu_freq() {
    local gpu_id=$1
    local freq=$2
    local label=$3
    echo ">>> [GPU Setup] Setting GPU $gpu_id to $label frequency..."
    sudo nvidia-smi -i "$gpu_id" -rgc > /dev/null 2>&1
    if [ "$label" != "default" ]; then
        sudo nvidia-smi -i "$gpu_id" -lgc "$freq"
    fi
}

update_config_qps() {
    local qps_target=$1
    echo ">>> [Config Setup] Updating config.json to poisson_lambda: $qps_target"
    if command -v jq >/dev/null 2>&1; then
        jq ".inputs.video_mme_input.poisson_lambda = $qps_target" "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
    else
        sed -i -E 's/"poisson_lambda": [0-9.]+/"poisson_lambda": '$qps_target'/' "$CONFIG_FILE"
    fi
}

run_pipeline() {
    local run_label=$1
    local experiment_output_dir="$MASTER_OUTPUT_DIR/$run_label"
    
    echo "------------------------------------------------"
    echo "Running Pipeline: $run_label"
    echo "Saving to: $experiment_output_dir"
    echo "------------------------------------------------"
    
    mkdir -p "$experiment_output_dir"
    EXPERIMENTS_SUBDIR="$experiment_output_dir/experiment_results"
    MONITOR_SUBDIR="$EXPERIMENTS_SUBDIR/monitoring_logs"
    cd "$DIR_PATH" || exit

    if [ -f "run_experiments.py" ]; then
        mkdir -p "$EXPERIMENTS_SUBDIR"
        PYTHONPATH="$PARENT_DIR" python3 run_experiments.py "$EXPERIMENTS_SUBDIR"
    fi

    if [ -f "$PROCESS_FOLDER/data_vis.py" ]; then
        DATAVIS_SUBDIR="$experiment_output_dir/plots"
        mkdir -p "$DATAVIS_SUBDIR"
        python3 "$PROCESS_FOLDER/data_vis.py" "$DATAVIS_SUBDIR" "$EXPERIMENTS_SUBDIR" --file-format "$FILE_FORMAT" --latency-nodes $LATENCY_NODES --value-nodes $VALUE_NODES --timeline-nodes $TIMELINE_NODES
    fi

    if [ -f "$PROCESS_FOLDER/plot_metrics.py" ]; then
        METRIC_PLOT_SUBDIR="$experiment_output_dir/metric_plots"
        mkdir -p "$METRIC_PLOT_SUBDIR"
        (cd "$PARENT_DIR" && python3 -m process_data.plot_metrics "$METRIC_PLOT_SUBDIR" "$MONITOR_SUBDIR" "$CONFIG_FILE")
    fi
}


echo "### STARTING FREQUENCY & QPS SWEEP ###"

for qps in "${QPS_VALUES[@]}"; do
    update_config_qps "$qps"

    for v_i in "${!FREQ_VALUES[@]}"; do
        vllm_val=${FREQ_VALUES[$v_i]}
        vllm_label=${FREQ_LABELS[$v_i]}
        
        for s_i in "${!FREQ_VALUES[@]}"; do
            stt_val=${FREQ_VALUES[$s_i]}
            stt_label=${FREQ_LABELS[$s_i]}

            set_gpu_freq $VLLM_GPU_ID "$vllm_val" "$vllm_label"
            set_gpu_freq $STT_GPU_ID "$stt_val" "$stt_label"
            
            # --- NEW: Inner loop for TRIALS ---
            for trial in $(seq 1 $NUM_TRIALS); do
                EXP_NAME="qps_${qps}_vllm_${vllm_label}_stt_${stt_label}_trial_${trial}"
                
                echo "========================================================"
                echo "PREPARING RUN: $EXP_NAME ($trial / $NUM_TRIALS)"
                echo "vLLM GPU ($VLLM_GPU_ID): $vllm_label ($vllm_val MHz)"
                echo "STT GPU  ($STT_GPU_ID): $stt_label ($stt_val MHz)"
                echo "========================================================"

                sleep 2
                start_services
                run_pipeline "$EXP_NAME"
                stop_services
            done
        done
    done
done

echo "--- Experiments Completed ---"
echo "Resetting Clocks..."
sudo nvidia-smi -rgc

echo "--- Generating Combined Heatmap Summary ---"

# Pass the qps list dynamically and specify a latency file if you want to isolate frame counts
if [ -f "$PROCESS_FOLDER/analyze_freq_experiments.py" ]; then
    python3 "$PROCESS_FOLDER/analyze_freq_experiments.py" \
        "$MASTER_OUTPUT_DIR" \
        --output "$MASTER_OUTPUT_DIR" \
        --power_file "*.json" \
        --qps_list "${QPS_VALUES[@]}" \
        --combined_summary
    
    echo "Done. Results saved in $MASTER_OUTPUT_DIR"
else
    echo "Warning: $PROCESS_FOLDER/analyze_freq_experiments.py not found. Skipping heatmaps."
fi