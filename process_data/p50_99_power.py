import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---

FILE_PATHS = [
    "/home/cerangel/comp-ai-bench/video-qa/3x3_results/0.2_freq_exp_results_2025-12-25_02-45-15/vllm_low_stt_low/experiment_results/monitoring_logs/processed_data/dcgmi-local-20251225-024633.csv",
    "/home/cerangel/comp-ai-bench/video-qa/3x3_results/0.2_freq_exp_results_2025-12-25_02-45-15/vllm_med_stt_low/experiment_results/monitoring_logs/processed_data/dcgmi-local-20251225-040836.csv",
    "/home/cerangel/comp-ai-bench/video-qa/3x3_results/0.2_freq_exp_results_2025-12-25_02-45-15/vllm_high_stt_low/experiment_results/monitoring_logs/processed_data/dcgmi-local-20251225-052703.csv",
]

LABELS = [
    "Low Frequency (300 MHz)", 
    "Medium Frequency (855 MHz)", 
    "High Frequency (1125 MHz)"
]

METRIC_COL = "POWER"
TIME_COL = "timestamp_offset" 
Y_AXIS_LABEL = "Power (W)"
X_AXIS_LABEL = "Time (s)"

# --- UPDATED FONT SIZES ---
FONT_SIZE_LABEL = 22   # Increased from 18
FONT_SIZE_TICK = 16    # Increased from 14
FONT_SIZE_LEGEND = 16  # Increased from 13

def plot_time_series_comparison(paths, labels):
    if len(paths) != len(labels):
        print("Error: The number of file paths must match the number of labels.")
        return

    # Create a single plot (compact size)
    plt.figure(figsize=(10, 6)) # Slight height increase to accommodate larger legend

    for file_path, label in zip(paths, labels):
        try:
            df = pd.read_csv(file_path)
            
            if METRIC_COL not in df.columns:
                print(f"Skipping {label}: Missing col {METRIC_COL}")
                continue

            # Handle Time Axis
            if TIME_COL in df.columns:
                x_data = df[TIME_COL]
            else:
                x_data = df.index

            y_data = df[METRIC_COL]

            # Calculate Stats for the Legend
            p50 = np.percentile(y_data, 50)
            p90 = np.percentile(y_data, 90)
            
            # Format label with stats
            legend_label = f"{label} | P50: {p50:.1f}W, P90: {p90:.1f}W"

            # Plot Line
            plt.plot(x_data, y_data, linewidth=1.5, alpha=0.85, label=legend_label)

        except Exception as e:
            print(f"Could not process {file_path}: {e}")

    # --- Formatting ---
    
    # Force axes to start at 0
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # Axis Labels
    plt.xlabel(X_AXIS_LABEL, fontsize=FONT_SIZE_LABEL)
    plt.ylabel(Y_AXIS_LABEL, fontsize=FONT_SIZE_LABEL)
    
    # Tick Size
    plt.xticks(fontsize=FONT_SIZE_TICK)
    plt.yticks(fontsize=FONT_SIZE_TICK)
    
    # Legend (Now includes P50/P90 stats)
    plt.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right', framealpha=0.95)
    
    # Reduce Whitespace
    plt.tight_layout()
    
    output_filename = "power_time_series_final.pdf"
    plt.savefig(output_filename)
    print(f"Success! Plot saved to: {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_time_series_comparison(FILE_PATHS, LABELS)