#!/bin/bash

# ==============================================================================
# vLLM Cache & GPU Utilization Sweep Benchmarking Script
# 
# Description: 
#   Automates benchmarking of a multimodal pipeline across different vLLM 
#   configurations (GPU Utilization limit and MM Cache Size). It launches 
#   services, executes Python-based load testing, and processes all metrics 
#   and visualizations (Latency, Hardware, and Cache usage).
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Path Resolution & Setup
# ------------------------------------------------------------------------------
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DIR_PATH="$SCRIPT_DIR"
PARENT_DIR=$( dirname "$SCRIPT_DIR" )

# ------------------------------------------------------------------------------
# 2. Service Configuration
# ------------------------------------------------------------------------------
# vLLM Settings
VLLM_START_SCRIPT="$PARENT_DIR/start_instance/start_vllm.sh"
MODEL="google/gemma-3-27b-it"
GPUS='"device=0,1"'
CPUS=""
TP_SIZE=2

# STT (Whisper) Settings
STT_START_SCRIPT="$PARENT_DIR/start_instance/start_whisper.sh"
STT_GPU_ID=1

# ------------------------------------------------------------------------------
# 3. Experiment Sweep Variables
# ------------------------------------------------------------------------------
RUNS_PER_CONFIG=1

# Test matrices: Iterate over GPU memory utilization and MM Cache sizes (in GB)
UTIL_VALUES=(0.5) 
CACHE_VALUES=(0 5) 

# ------------------------------------------------------------------------------
# 4. Data Processing & Output Configuration
# ------------------------------------------------------------------------------
PROCESS_FOLDER="$PARENT_DIR/process_data"
CONFIG_FILE="$DIR_PATH/config.json"

# Metrics to extract during the data visualization phase
FILE_FORMAT="frames-*_transcription-*.csv"
LATENCY_NODES="encode_videos speech_to_text multimodal_model end_to_end end_to_end_load_gen"
VALUE_NODES="prompt_tokens total_tokens"
TIMELINE_NODES="encode_videos speech_to_text multimodal_model"

# Create a unique master directory for this batch of experiments
BATCH_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
MASTER_OUTPUT_DIR="$SCRIPT_DIR/cache_exp_results_$BATCH_TIMESTAMP"
mkdir -p "$MASTER_OUTPUT_DIR"
echo ">>> Master output directory created at: $MASTER_OUTPUT_DIR"

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
    
    sudo docker stop vllm_experiment_container 2>/dev/null || true
    sudo docker rm vllm_experiment_container 2>/dev/null || true

    sudo docker stop murakkab-whisper 2>/dev/null || true
    sudo docker rm murakkab-whisper 2>/dev/null || true
}

# Boots up the required ML services and waits for them to stabilize
start_services() {
    local util=$1
    local mm_cache=$2
    
    echo "------------------------------------------------"
    echo "Starting Services (Util: $util, Cache: ${mm_cache}GB)"
    echo "------------------------------------------------"
    
    echo "Launching vLLM..."
    bash "$VLLM_START_SCRIPT" "$GPUS" "$MODEL" "$TP_SIZE" "$CPUS" "$util" "$mm_cache"

    echo "Launching STT (Whisper)..."
    bash "$STT_START_SCRIPT" gpu "$STT_GPU_ID"

    # Wait for both to be fully active
    wait_for_service "http://localhost:8000/health" "vLLM"
    
    echo "Waiting 10s for STT to stabilize..."
    sleep 10
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

    # 4. Process vLLM KV and Multi-Modal cache metrics
    if [ -f "$PROCESS_FOLDER/mm_kv_plotter.py" ]; then
        CACHE_PLOT_SUBDIR="$experiment_output_dir/cache_plots"
        mkdir -p "$CACHE_PLOT_SUBDIR"
        
        # Pointing to the exact filename your python script generates
        VLLM_LOG_FILE="$MONITOR_SUBDIR/vllm_cache_log.csv"
        
        if [ -f "$VLLM_LOG_FILE" ]; then
            # Execute exactly like the hardware metrics script
            (cd "$PARENT_DIR" && python3 -m process_data.mm_kv_plotter "$VLLM_LOG_FILE" --output "${run_label}_cache.pdf" --output-dir "$CACHE_PLOT_SUBDIR")
        else
            # Fallback: If vLLM drops the log in the root execution folder instead
            if [ -f "./vllm_cache_log.csv" ]; then
                (cd "$PARENT_DIR" && python3 -m process_data.mm_kv_plotter "./vllm_cache_log.csv" --output "${run_label}_cache.pdf" --output-dir "$CACHE_PLOT_SUBDIR")
                mv "./vllm_cache_log.csv" "$CACHE_PLOT_SUBDIR/"
            else
                echo ">>> Warning: vllm_cache_log.csv not found in $MONITOR_SUBDIR or root. Skipping cache plots."
            fi
        fi
    fi
}

# ==============================================================================
# Main Execution Flow
# ==============================================================================

echo "### STARTING CONFIGURATION SWEEP ###"

# Clean any lingering instances before we start the loops
stop_services

# Loop 1: Iterate through GPU utilization limits
for util in "${UTIL_VALUES[@]}"; do
    
    # Loop 2: Iterate through Multimodal Cache sizes
    for mm_cache in "${CACHE_VALUES[@]}"; do
    
        # Loop 3: Execute multiple trials per configuration
        for trial in $(seq 1 $RUNS_PER_CONFIG); do
            EXP_NAME="util_${util}_cache_${mm_cache}GB_trial_${trial}"
            
            echo "========================================================"
            echo "PREPARING RUN: $EXP_NAME ($trial / $RUNS_PER_CONFIG)"
            echo "========================================================"

            start_services "$util" "$mm_cache" # Boot up fresh containers
            run_pipeline "$EXP_NAME"           # Generate load and collect data
            stop_services                      # Tear down to prevent memory leaks between runs
            
            sleep 5
        done
    done
done

echo "### All configurations and runs completed successfully. ###"
echo "### All data saved to: $MASTER_OUTPUT_DIR ###"