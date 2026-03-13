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

def parse_qps_from_path(path):
    """
    Extracts QPS from folder names like '0.1_run1', 'qps_0.1', or '0.1'.
    """
    path_parts = path.split(os.sep)
    for part in reversed(path_parts):
        # Priority 1: Start with number + underscore (e.g., "0.1_run1")
        match_start = re.search(r"^(\d+\.?\d*)_", part)
        if match_start: return float(match_start.group(1))

        # Priority 2: "qps_" prefix (e.g., "qps_0.1")
        match_qps = re.search(r"qps_(\d+\.?\d*)", part, re.IGNORECASE)
        if match_qps: return float(match_qps.group(1))

        # Priority 3: Exact number (e.g., "0.1")
        try:
            val = float(part)
            return val
        except ValueError:
            continue
    return "Unknown"

def load_data(base_dirs, file_pattern):
    data_records = []
    if isinstance(base_dirs, str): base_dirs = [base_dirs]

    print(f"Scanning directories: {base_dirs}")

    for base_dir in base_dirs:
        search_path = os.path.join(base_dir, "**", file_pattern)
        files = glob.glob(search_path, recursive=True)

        if not files:
            continue

        for file_path in files:
            # 1. Parse Frequencies
            vllm_label, stt_label = None, None
            for part in file_path.split(os.sep):
                v, s = parse_folder_name(part)
                if v and s:
                    vllm_label, stt_label = v, s
                    break
            
            if not vllm_label: continue

            # 2. Parse QPS
            qps_val = parse_qps_from_path(file_path)

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                record = {
                    "qps": qps_val,
                    "vllm_label": vllm_label,
                    "stt_label": stt_label,
                    "vllm_freq": FREQ_MAP.get(vllm_label, 0),
                    "stt_freq": FREQ_MAP.get(stt_label, 0),
                    "source_file": file_path
                }
                
                has_data = False
                for m in METRICS:
                    if m in data:
                        record[m] = data[m]
                        has_data = True
                
                if has_data: data_records.append(record)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return pd.DataFrame(data_records)

def generate_best_config_table(df, output_dir, optimize_metric="ENERGY"):
    if optimize_metric not in df.columns:
        print(f"Metric '{optimize_metric}' not found.")
        return

    # Aggregate runs (calculate mean across runs for same settings)
    group_cols = ["qps", "vllm_freq", "stt_freq", "vllm_label", "stt_label"]
    df_agg = df.groupby(group_cols)[optimize_metric].agg(['mean', 'std']).reset_index()

    # Find Best Config
    best_indices = df_agg.groupby("qps")['mean'].idxmin()
    best_df = df_agg.loc[best_indices].sort_values(by="qps")

    best_df['result'] = best_df.apply(lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}", axis=1)
    final_table = best_df[["qps", "vllm_freq", "stt_freq", "vllm_label", "stt_label", "result"]]
    final_table = final_table.rename(columns={"result": f"{optimize_metric} (Mean ± Std)"})

    print(f"\n=== Best Configurations per QPS (Optimizing {optimize_metric}) ===")
    print(final_table.to_string(index=False))
    print("==============================================================\n")

    csv_path = os.path.join(output_dir, f"best_configs_by_{optimize_metric}.csv")
    final_table.to_csv(csv_path, index=False)
    print(f"Table saved to: {csv_path}")

def plot_heatmaps_per_qps(df, output_dir, colormap="viridis"):
    """
    Generates a separate PDF heatmap for each QPS value found in the data.
    """
    if df.empty: return
    valid_metrics = [m for m in METRICS if m in df.columns]
    
    # Identify all unique QPS values
    unique_qps = df['qps'].unique()
    print(f"Found QPS values: {unique_qps}")

    for qps in unique_qps:
        print(f"Generating heatmap for QPS: {qps}")
        
        # Filter data for this QPS only
        sub_df = df[df['qps'] == qps]
        
        # Aggregate multiple runs (e.g., run1, run2) for this specific QPS
        grouped = sub_df.groupby(["vllm_freq", "stt_freq"])[valid_metrics]
        df_mean = grouped.mean()
        df_std = grouped.std().fillna(0) # 0 if only 1 run

        num_plots = len(valid_metrics)
        cols = 2
        rows = (num_plots + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 7 * rows))
        axes = axes.flatten()

        for i, metric in enumerate(valid_metrics):
            pivot_mean = df_mean.reset_index().pivot(index="vllm_freq", columns="stt_freq", values=metric)
            pivot_std = df_std.reset_index().pivot(index="vllm_freq", columns="stt_freq", values=metric)
            
            pivot_mean.sort_index(ascending=True, inplace=True)
            pivot_mean.sort_index(axis=1, ascending=True, inplace=True)
            pivot_std.sort_index(ascending=True, inplace=True)
            pivot_std.sort_index(axis=1, ascending=True, inplace=True)

            annot_labels = pivot_mean.map(lambda x: f"{x:.2f}") + "\n±" + pivot_std.map(lambda x: f"{x:.2f}")
            
            ax = axes[i]
            sns.heatmap(pivot_mean, annot=annot_labels, fmt="", cmap=colormap, ax=ax, annot_kws={"size": 10})
            
            # Title includes QPS
            ax.set_title(f"QPS {qps}: {TITLES.get(metric, metric)}")
            ax.set_xlabel("STT Freq (MHz)")
            ax.set_ylabel("vLLM Freq (MHz)")
            ax.invert_yaxis()

        for j in range(i + 1, len(axes)): axes[j].axis('off')
        
        plt.suptitle(f"Heatmap for QPS: {qps}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit suptitle
        
        output_filename = f"heatmap_qps_{qps}_{colormap}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path)
        plt.close(fig) # Close memory
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", nargs='+', type=str, help="List of experiment directories")
    parser.add_argument("--filename", type=str, default="*_summary.json", help="Glob pattern for JSON files")
    parser.add_argument("--cmap", type=str, default="YlOrRd", help="Colormap for heatmaps")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    parser.add_argument("--optimize", type=str, default="ENERGY", choices=METRICS, help="Metric to minimize")
    
    args = parser.parse_args()
    if not os.path.exists(args.output): os.makedirs(args.output)

    df = load_data(args.input_dirs, args.filename)
    
    if not df.empty:
        # 1. Best Config Table
        generate_best_config_table(df, args.output, optimize_metric=args.optimize)
        
        # 2. Separate Heatmaps for each QPS
        plot_heatmaps_per_qps(df, args.output, colormap=args.cmap)
    else:
        print("No valid data found.")