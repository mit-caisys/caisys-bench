SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DIR_PATH="$SCRIPT_DIR"


PARENT_DIR=$( dirname "$SCRIPT_DIR" )


MONITORING_FOLDER="$PARENT_DIR/monitoring"
PROCESS_FOLDER="$PARENT_DIR/process_data"


FILE_FORMAT="frames-*_transcription-*.csv"

# SORTING_METHOD= ""


LATENCY_NODES="encode_videos speech_to_text multimodal_model end_to_end end_to_end_load_gen"
VALUE_NODES="prompt_tokens total_tokens"
TIMELINE_NODES="encode_videos speech_to_text multimodal_model"

echo $DIR_PATH

if [ ! -d "$DIR_PATH" ]; then
  echo "Error: Could not determine the script's directory."
  exit 1
fi

cd "$DIR_PATH" || exit

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR="$SCRIPT_DIR/results_$TIMESTAMP"

# Create the directory
mkdir -p "$OUTPUT_DIR"

echo "All results will be saved in: $OUTPUT_DIR"

echo "--- Starting Experiments---"

EXPERIMENTS_SUBDIR="$OUTPUT_DIR/experiment_results"
MONITOR_SUBDIR="$EXPERIMENTS_SUBDIR/monitoring_logs"

CONFIG_FILE="$DIR_PATH/config.json"

if [ -f "run_experiments.py" ]; then

  mkdir -p "$EXPERIMENTS_SUBDIR"

  echo "Running run_experiments..."
  PYTHONPATH="$PARENT_DIR" python3 run_experiments.py "$EXPERIMENTS_SUBDIR"
  echo "Finished run_experiments.py."
else
  echo "Warning: run_experiments.py not found. Skipping."
fi

if [ -f "$PROCESS_FOLDER/data_vis.py" ]; then

  DATAVIS_SUBDIR="$OUTPUT_DIR/plots"
  mkdir -p "$DATAVIS_SUBDIR"

  echo "Visualizing data..."
  python3 "$PROCESS_FOLDER/data_vis.py"\
   "$DATAVIS_SUBDIR" \
   "$EXPERIMENTS_SUBDIR" \
   --file-format "$FILE_FORMAT" \
   --latency-nodes $LATENCY_NODES \
   --value-nodes $VALUE_NODES \
   --timeline-nodes $TIMELINE_NODES
  echo "Finished data_vis.py."
else
  echo "Warning: data_vis.py not found. Skipping."
fi

if [ -f "$PROCESS_FOLDER/plot_metrics.py" ]; then

  METRIC_PLOT_SUBDIR="$OUTPUT_DIR/metric_plots"
  mkdir -p "$METRIC_PLOT_SUBDIR"

  echo "Visualizing data metrics..."
  # python3 "$PROCESS_FOLDER/plot_metrics.py" "$METRIC_PLOT_SUBDIR" "$MONITOR_SUBDIR" "$CONFIG_FILE"
  (cd "$PARENT_DIR" && python3 -m process_data.plot_metrics "$METRIC_PLOT_SUBDIR" "$MONITOR_SUBDIR" "$CONFIG_FILE")
  echo "Finished plot_metrics.py."
else
  echo "Warning: plot_metrics.py not found. Skipping."
fi

echo "--- All scripts have been executed. ---"

cd - > /dev/null
