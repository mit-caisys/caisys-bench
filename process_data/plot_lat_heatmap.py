import os
import glob
import re
import argparse
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

PERCENTILES = [50, 90, 95, 99]

def parse_folder_name(folder_name):
    """Parses 'vllm_low_stt_high' into ('low', 'high')"""
    match = re.search(r"vllm_([a-zA-Z0-9]+)_stt_([a-zA-Z0-9]+)", folder_name)
    if match:
        return match.group(1), match.group(2)
    return None, None

def load_data(base_dir, file_pattern):
    data_records = []
    search_path = os.path.join(base_dir, "**", file_pattern)
    files = glob.glob(search_path, recursive=True)

    if not files:
        print(f"No files found matching: {file_pattern} in {base_dir}")
        return pd.DataFrame()

    print(f"Found {len(files)} CSV files. Processing...")

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
            df = pd.read_csv(file_path)
            
            # Check for required columns
            if 'end_to_end_end' in df.columns and 'end_to_end_start' in df.columns:
                latencies = df['end_to_end_end'] - df['end_to_end_start']
                
                stats = {
                    "vllm_label": vllm_label,
                    "stt_label": stt_label,
                    "vllm_freq": FREQ_MAP.get(vllm_label, 0),
                    "stt_freq": FREQ_MAP.get(stt_label, 0)
                }
                
                for p in PERCENTILES:
                    stats[f"p{p}"] = latencies.quantile(p / 100.0)
                
                data_records.append(stats)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return pd.DataFrame(data_records)

def plot_heatmaps(df, output_dir):
    if df.empty:
        print("Dataframe is empty, skipping plots.")
        return

    df = df.sort_values(by=["vllm_freq", "stt_freq"])

    num_plots = len(PERCENTILES)
    cols = 2
    rows = (num_plots + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    axes = axes.flatten()

    for i, p in enumerate(PERCENTILES):
        col_name = f"p{p}"
        pivot_table = df.pivot(index="vllm_freq", columns="stt_freq", values=col_name)
        
        pivot_table.sort_index(ascending=True, inplace=True) 
        pivot_table.sort_index(axis=1, ascending=True, inplace=True)

        ax = axes[i]
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, cbar_kws={'label': 'Latency (s)'})
        
        ax.set_title(f"End-to-End Latency (P{p})")
        ax.set_xlabel("STT GPU Frequency (MHz)")
        ax.set_ylabel("vLLM GPU Frequency (MHz)")
        ax.invert_yaxis()

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "latency_heatmap_matrix.pdf")
    plt.savefig(output_path)
    print(f"Heatmap saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--filename", type=str, default="frames-*.csv")
    
    args = parser.parse_args()
    
    df = load_data(args.input_dir, args.filename)
    if not df.empty:
        plot_heatmaps(df, args.input_dir)
    else:
        print("No data found.")