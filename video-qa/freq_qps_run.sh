#!/bin/bash

# ==============================================================================
# GPU Frequency & QPS Sweep Benchmarking Script
# 
# Description: 
#   Automates benchmarking of a multimodal pipeline (vLLM + Whisper STT) across 
#   different server loads (QPS) and GPU clock frequencies. It launches services, 
#   modifies configurations on the fly, executes trials, processes metrics, and 
#   generates visualization heatmaps.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Path Resolution & Setup
# ------------------------------------------------------------------------------
# Get the absolute path of the directory containing this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DIR_PATH="$SCRIPT_DIR"
PARENT_DIR=$( dirname "$SCRIPT_DIR" )

# ------------------------------------------------------------------------------
# 2. Service Configuration
# ------------------------------------------------------------------------------
# vLLM (Large Language Model) Settings
VLLM_START_SCRIPT="$PARENT_DIR/start_instance/start_vllm.sh"
VLLM_GPU_ID=0
MODEL="google/gemma-3-27b-it"
GPUS='"device=0"'
CPUS=""
TP_SIZE=1

# STT (Speech-to-Text / Whisper) Settings
STT_START_SCRIPT="$PARENT_DIR/start_instance/start_whisper.sh"
STT_GPU_ID=1

# ------------------------------------------------------------------------------
# 3. Experiment Variables (Frequencies & Modes)
# ------------------------------------------------------------------------------
# Defined GPU core frequencies (in MHz) to test
FREQ_MIN=300
FREQ_LOW=570
FREQ_MID=855
FREQ_HIGH=1125
FREQ_MAX=1410

# Parse command-line arguments to toggle between quick testing and full data collection
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

# Target Queries Per Second (QPS) to simulate different traffic loads
QPS_VALUES=(0.2 0.4 0.6)

# ------------------------------------------------------------------------------
# 4. Data Processing & Output Configuration
# ------------------------------------------------------------------------------
MONITORING_FOLDER="$PARENT_DIR/monitoring"
PROCESS_FOLDER="$PARENT_DIR/process_data"
CONFIG_FILE="$DIR_PATH/config.json"

# Metrics to extract during the data visualization phase
FILE_FORMAT="frames-*_transcription-*.csv"
LATENCY_NODES="encode_videos speech_to_text multimodal_model end_to_end end_to_end_load_gen"
VALUE_NODES="prompt_tokens total_tokens"
TIMELINE_NODES="encode_videos speech_to_text multimodal_model"

# Create a unique master directory for this batch of experiments
BATCH_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
MASTER_OUTPUT_DIR="$SCRIPT_DIR/freq_exp_results_$BATCH_TIMESTAMP"
mkdir -p "$MASTER_OUTPUT_DIR"

# ==============================================================================
# Helper Functions
# ==============================================================================

# Blocks execution until a specific service endpoint returns a successful HTTP status
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

# Gracefully stops and removes any running Docker containers from previous runs
stop_services() {
    echo "------------------------------------------------"
    echo "Stopping Services..."
    echo "------------------------------------------------"
    
    # Clean up vLLM container
    if [ "$(sudo docker ps -q -f name=vllm_experiment_container)" ]; then
        sudo docker stop vllm_experiment_container
    fi
    sudo docker rm vllm_experiment_container 2>/dev/null || true

    # Clean up STT container
    if [ "$(sudo docker ps -q -f name=murakkab-whisper)" ]; then
        sudo docker stop murakkab-whisper
    fi
    sudo docker rm murakkab-whisper 2>/dev/null || true
    
    echo "All services stopped."
}

# Boots up the required ML services and waits for them to stabilize
start_services() {
    echo "------------------------------------------------"
    echo "PHASE 0: Starting Services"
    echo "------------------------------------------------"
    
    echo "Launching vLLM on GPU $VLLM_GPU_ID..."
    bash "$VLLM_START_SCRIPT" "$GPUS" "$MODEL" "$TP_SIZE" "$CPUS"

    echo "Launching STT (Whisper) on GPU $STT_GPU_ID..."
    bash "$STT_START_SCRIPT" gpu "$STT_GPU_ID"

    # Ensure vLLM is fully initialized before proceeding
    wait_for_service "http://localhost:8000/health" "vLLM"
    
    # Whisper doesn't have a health endpoint in this setup, so we use a hard sleep
    echo "Waiting 10s for STT to stabilize..."
    sleep 10
    echo "Services started successfully."
}

# Uses nvidia-smi to lock the GPU to a specific clock frequency
set_gpu_freq() {
    local gpu_id=$1
    local freq=$2
    local label=$3
    
    echo ">>> [GPU Setup] Setting GPU $gpu_id to $label frequency..."
    
    # First, reset to default clocks to clear any previous locks
    sudo nvidia-smi -i "$gpu_id" -rgc > /dev/null 2>&1
    
    # Apply the new frequency lock unless "default" is specified
    if [ "$label" != "default" ]; then
        sudo nvidia-smi -i "$gpu_id" -lgc "$freq"
    fi
}

# Dynamically updates the load generation rate in the JSON configuration
update_config_qps() {
    local qps_target=$1
    echo ">>> [Config Setup] Updating config.json to poisson_lambda: $qps_target"
    
    # Try using 'jq' for safe JSON manipulation; fallback to 'sed' if jq is unavailable
    if command -v jq >/dev/null 2>&1; then
        jq ".inputs.video_mme_input.poisson_lambda = $qps_target" "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
    else
        sed -i -E 's/"poisson_lambda": [0-9.]+/"poisson_lambda": '$qps_target'/' "$CONFIG_FILE"
    fi
}

# Executes the Python benchmark suite and all subsequent data visualization scripts
run_pipeline() {
    local run_label=$1
    local experiment_output_dir="$MASTER_OUTPUT_DIR/$run_label"
    
    echo "------------------------------------------------"
    echo "Running Pipeline: $run_label"
    echo "Saving to: $experiment_output_dir"
    echo "------------------------------------------------"
    
    # Setup directories for this specific run
    mkdir -p "$experiment_output_dir"
    EXPERIMENTS_SUBDIR="$experiment_output_dir/experiment_results"
    MONITOR_SUBDIR="$EXPERIMENTS_SUBDIR/monitoring_logs"
    cd "$DIR_PATH" || exit

    # 1. Run the main load test / benchmark
    if [ -f "run_experiments.py" ]; then
        mkdir -p "$EXPERIMENTS_SUBDIR"
        PYTHONPATH="$PARENT_DIR" python3 run_experiments.py "$EXPERIMENTS_SUBDIR"
    fi

    # 2. Process latency and timeline data into visualizations
    if [ -f "$PROCESS_FOLDER/data_vis.py" ]; then
        DATAVIS_SUBDIR="$experiment_output_dir/plots"
        mkdir -p "$DATAVIS_SUBDIR"
        python3 "$PROCESS_FOLDER/data_vis.py" "$DATAVIS_SUBDIR" "$EXPERIMENTS_SUBDIR" --file-format "$FILE_FORMAT" --latency-nodes $LATENCY_NODES --value-nodes $VALUE_NODES --timeline-nodes $TIMELINE_NODES
    fi

    # 3. Process hardware monitoring metrics (e.g., power draw, utilization)
    if [ -f "$PROCESS_FOLDER/plot_metrics.py" ]; then
        METRIC_PLOT_SUBDIR="$experiment_output_dir/metric_plots"
        mkdir -p "$METRIC_PLOT_SUBDIR"
        (cd "$PARENT_DIR" && python3 -m process_data.plot_metrics "$METRIC_PLOT_SUBDIR" "$MONITOR_SUBDIR" "$CONFIG_FILE")
    fi
}

# ==============================================================================
# Main Execution Flow
# ==============================================================================

echo "### STARTING FREQUENCY & QPS SWEEP ###"

# Loop 1: Iterate through each target server load (QPS)
for qps in "${QPS_VALUES[@]}"; do
    update_config_qps "$qps"

    # Loop 2: Iterate through vLLM GPU frequencies
    for v_i in "${!FREQ_VALUES[@]}"; do
        vllm_val=${FREQ_VALUES[$v_i]}
        vllm_label=${FREQ_LABELS[$v_i]}
        
        # Loop 3: Iterate through STT GPU frequencies
        for s_i in "${!FREQ_VALUES[@]}"; do
            stt_val=${FREQ_VALUES[$s_i]}
            stt_label=${FREQ_LABELS[$s_i]}

            # Apply the hardware locks
            set_gpu_freq $VLLM_GPU_ID "$vllm_val" "$vllm_label"
            set_gpu_freq $STT_GPU_ID "$stt_val" "$stt_label"
            
            # Loop 4: Execute multiple trials for statistical reliability
            for trial in $(seq 1 $NUM_TRIALS); do
                EXP_NAME="qps_${qps}_vllm_${vllm_label}_stt_${stt_label}_trial_${trial}"
                
                echo "========================================================"
                echo "PREPARING RUN: $EXP_NAME ($trial / $NUM_TRIALS)"
                echo "vLLM GPU ($VLLM_GPU_ID): $vllm_label ($vllm_val MHz)"
                echo "STT GPU  ($STT_GPU_ID): $stt_label ($stt_val MHz)"
                echo "========================================================"

                sleep 2
                start_services        # Boot up fresh containers
                run_pipeline "$EXP_NAME" # Generate load and collect data
                stop_services         # Tear down to prevent caching/memory leaks between runs
            done
        done
    done
done

# ==============================================================================
# Cleanup & Post-Processing
# ==============================================================================

echo "--- Experiments Completed ---"
echo "Resetting Clocks..."
sudo nvidia-smi -rgc # Remove all GPU frequency locks, returning to default power management

echo "--- Generating Combined Heatmap Summary ---"

# Aggregate all the individual trial data into final summary heatmaps
if [ -f "$PROCESS_FOLDER/analyze_freq_experiments.py" ]; then
    python3 "$PROCESS_FOLDER/analyze_freq_experiments.py" \
        "$MASTER_OUTPUT_DIR" \
        --output "$MASTER_OUTPUT_DIR" \
        --power_file "*_summary.json" \
        --qps_list "${QPS_VALUES[@]}" \
        --combined_summary
    
    echo "Done. Results saved in $MASTER_OUTPUT_DIR"
else
    echo "Warning: $PROCESS_FOLDER/analyze_freq_experiments.py not found. Skipping heatmaps."
fi