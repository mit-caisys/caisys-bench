
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


FREQ_VALUES=($FREQ_MIN $FREQ_LOW $FREQ_MID $FREQ_HIGH $FREQ_MAX)
FREQ_LABELS=("min" "low" "mid" "high" "max")

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
    
    # Stop vLLM
    if [ "$(sudo docker ps -q -f name=vllm_experiment_container)" ]; then
        echo "Stopping vLLM container..."
        sudo docker stop vllm_experiment_container
    fi
    sudo docker rm vllm_experiment_container 2>/dev/null || true

    # Stop STT
    if [ "$(sudo docker ps -q -f name=murakkab-whisper)" ]; then
        echo "Stopping STT container..."
        sudo docker stop murakkab-whisper
    fi
    sudo docker rm murakkab-whisper 2>/dev/null || true
    
    echo "All services stopped."
}

start_services() {
    echo "------------------------------------------------"
    echo "PHASE 0: Starting Services"
    echo "------------------------------------------------"

    if [ ! -f "$VLLM_START_SCRIPT" ]; then
        echo "Error: $VLLM_START_SCRIPT not found!"
        exit 1
    fi
    if [ ! -f "$STT_START_SCRIPT" ]; then
        echo "Error: $STT_START_SCRIPT not found!"
        exit 1
    fi

    # 1. Start vLLM
    echo "Launching vLLM on GPU $VLLM_GPU_ID..."
    bash "$VLLM_START_SCRIPT" \
        "$GPUS" \
        "$MODEL" \
        "$TP_SIZE" \
        "$CPUS"

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
    
    if [ "$label" == "default" ]; then
        echo "    Reset to default clocks."
    else
        sudo nvidia-smi -i "$gpu_id" -lgc "$freq"
        echo "    Locked clocks to $freq MHz."
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
    else
        echo "Error: run_experiments.py not found."
    fi

    if [ -f "$PROCESS_FOLDER/data_vis.py" ]; then
        DATAVIS_SUBDIR="$experiment_output_dir/plots"
        mkdir -p "$DATAVIS_SUBDIR"
        python3 "$PROCESS_FOLDER/data_vis.py"\
         "$DATAVIS_SUBDIR" \
         "$EXPERIMENTS_SUBDIR" \
         --file-format "$FILE_FORMAT" \
         --latency-nodes $LATENCY_NODES \
         --value-nodes $VALUE_NODES \
         --timeline-nodes $TIMELINE_NODES
    fi

    if [ -f "$PROCESS_FOLDER/plot_metrics.py" ]; then
        METRIC_PLOT_SUBDIR="$experiment_output_dir/metric_plots"
        mkdir -p "$METRIC_PLOT_SUBDIR"
        (cd "$PARENT_DIR" && python3 -m process_data.plot_metrics "$METRIC_PLOT_SUBDIR" "$MONITOR_SUBDIR" "$CONFIG_FILE")
    fi
}


echo "### STARTING FREQUENCY SWEEP ###"

for v_i in "${!FREQ_VALUES[@]}"; do
    vllm_val=${FREQ_VALUES[$v_i]}
    vllm_label=${FREQ_LABELS[$v_i]}
    
    for s_i in "${!FREQ_VALUES[@]}"; do
        stt_val=${FREQ_VALUES[$s_i]}
        stt_label=${FREQ_LABELS[$s_i]}

        EXP_NAME="vllm_${vllm_label}_stt_${stt_label}"
        echo "========================================================"
        echo "PREPARING RUN: $EXP_NAME"
        echo "vLLM GPU ($VLLM_GPU_ID): $vllm_label ($vllm_val MHz)"
        echo "STT GPU  ($STT_GPU_ID): $stt_label ($stt_val MHz)"
        echo "========================================================"

        set_gpu_freq $VLLM_GPU_ID "$vllm_val" "$vllm_label"
        set_gpu_freq $STT_GPU_ID "$stt_val" "$stt_label"
        
        sleep 2
        start_services

        run_pipeline "$EXP_NAME"

        stop_services
    done
done

echo "--- Experiments Completed ---"
echo "Resetting Clocks..."
sudo nvidia-smi -rgc


echo "--- Generating Heatmap Summary ---"

if [ -f "$PROCESS_FOLDER/plot_lat_heatmap.py" ]; then
    python3 "$PROCESS_FOLDER/plot_lat_heatmap.py" \
        "$MASTER_OUTPUT_DIR" \
        --filename "$FILE_FORMAT"
    
    echo "Done. Results saved in $MASTER_OUTPUT_DIR"
else
    echo "Warning: $PROCESS_FOLDER/plot_lat_heatmap.py not found. Skipping heatmaps."
fi

if [ -f "$PROCESS_FOLDER/plot_power_freq.py" ]; then
    python3 "$PROCESS_FOLDER/plot_power_freq.py" \
        "$MASTER_OUTPUT_DIR"
    
    echo "Done. Results saved in $MASTER_OUTPUT_DIR"
else
    echo "Warning: $PROCESS_FOLDER/plot_power_freq.py not found. Skipping heatmaps."
fi