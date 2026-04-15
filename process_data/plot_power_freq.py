import os
import glob
import re
import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FREQ_MAP = {
    "min": 300,
    "low": 570,
    "mid": 855,
    "high": 1125,
    "max": 1410,
    "default": 0
}

# Metrics to extract from the JSON
METRICS = ["POWER_P50", "POWER_P90", "POWER_P99", "ENERGY"]
TITLES = {
    "POWER_P50": "Power Consumption (P50) [W]",
    "POWER_P90": "Power Consumption (P90) [W]",
    "POWER_P99": "Power Consumption (P99) [W]",
    "ENERGY": "Total Energy [kWh]" 
}

def parse_folder_name(folder_name):
    """Parses 'vllm_low_stt_high' into ('low', 'high')"""
    match = re.search(r"vllm_([a-zA-Z0-9]+)_stt_([a-zA-Z0-9]+)", folder_name)
    if match:
        return match.group(1), match.group(2)
    return None, None

def load_data(base_dir, file_pattern):
    data_records = []
    # Search recursively for the JSON files
    search_path = os.path.join(base_dir, "**", file_pattern)
    files = glob.glob(search_path, recursive=True)

    if not files:
        print(f"No files found matching: {file_pattern} in {base_dir}")
        return pd.DataFrame()

    print(f"Found {len(files)} JSON files. Processing...")

    for file_path in files:
        path_parts = file_path.split(os.sep)
        vllm_label, stt_label = None, None
        
        for part in path_parts:
            v, s = parse_folder_name(part)
            if v and s:
                vllm_label, stt_label = v, s
                break
        
        if not vllm_label:
            continue

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            record = {
                "vllm_label": vllm_label,
                "stt_label": stt_label,
                "vllm_freq": FREQ_MAP.get(vllm_label, 0),
                "stt_freq": FREQ_MAP.get(stt_label, 0)
            }
            
            has_data = False
            for m in METRICS:
                if m in data:
                    record[m] = data[m]
                    has_data = True
            
            if has_data:
                data_records.append(record)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return pd.DataFrame(data_records)

def plot_heatmaps(df, output_dir, colormap="viridis"):
    if df.empty:
        print("Dataframe is empty, skipping plots.")
        return

    df = df.sort_values(by=["vllm_freq", "stt_freq"])

    valid_metrics = [m for m in METRICS if m in df.columns]

    if not valid_metrics:
        print("No valid metrics found in data.")
        return

    num_plots = len(valid_metrics)
    cols = 2
    rows = (num_plots + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    axes = axes.flatten()

    for i, metric in enumerate(valid_metrics):
        pivot_table = df.pivot_table(index="vllm_freq", columns="stt_freq", values=metric, aggfunc="mean")
        
        pivot_table.sort_index(ascending=True, inplace=True) 
        pivot_table.sort_index(axis=1, ascending=True, inplace=True)

        ax = axes[i]
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap=colormap, ax=ax)
        
        ax.set_title(TITLES.get(metric, metric))
        ax.set_xlabel("STT GPU Frequency (MHz)")
        ax.set_ylabel("vLLM GPU Frequency (MHz)")
        ax.invert_yaxis()

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"power_energy_heatmap_matrix_{colormap}.pdf")
    plt.savefig(output_path)
    print(f"Power/Energy Heatmap saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Path to the master results directory")
    parser.add_argument("--filename", type=str, default="*_summary.json", help="Glob pattern for JSON files")
    parser.add_argument("--cmap", type=str, default="YlOrRd", help="Colormap (default: YlOrRd for power)")
    
    args = parser.parse_args()
    
    df = load_data(args.input_dir, args.filename)
    if not df.empty:
        plot_heatmaps(df, args.input_dir, colormap=args.cmap)
    else:
        print("No data found.")