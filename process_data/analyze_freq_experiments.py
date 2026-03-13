import os
import glob
import re
import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

FREQ_MAP = {
    "min": 300,
    "low": 570,
    "mid": 855,
    "high": 1125,
    "max": 1410,
    "default": 0
}

POWER_METRICS = ["POWER_P50", "POWER_P90", "POWER_P99", "ENERGY"]
LATENCY_PERCENTILES = [50, 90, 95, 99]
LATENCY_METRICS = [f"p{p}" for p in LATENCY_PERCENTILES]

TITLES = {
    "POWER_P50": "P50 Power (W)",
    "POWER_P90": "P90 Power (W)",
    "POWER_P99": "P99 Power (W)",
    "ENERGY": "Energy (kWh)",
    "p50": "P50 Latency (s)",
    "p90": "P90 Latency (s)",
    "p95": "P95 Latency (s)",
    "p99": "P99 Latency (s)"
}

def parse_folder_name(folder_name):
    match = re.search(r"vllm_([a-zA-Z0-9]+)_stt_([a-zA-Z0-9]+)", folder_name)
    if match:
        return match.group(1), match.group(2)
    return None, None

def parse_qps_from_path(path):
    path_parts = path.split(os.sep)
    for part in reversed(path_parts):
        match_start = re.search(r"^(\d+\.?\d*)_", part)
        if match_start: return float(match_start.group(1))
        match_qps = re.search(r"qps_(\d+\.?\d*)", part, re.IGNORECASE)
        if match_qps: return float(match_qps.group(1))
        try:
            val = float(part)
            return val
        except ValueError:
            continue
    return "Unknown"

def load_power_data(base_dirs, file_pattern):
    data_records = []
    if isinstance(base_dirs, str): base_dirs = [base_dirs]
    print(f"\n--- Scanning for Power Data ({file_pattern}) ---")
    
    for base_dir in base_dirs:
        search_path = os.path.join(base_dir, "**", file_pattern)
        files = glob.glob(search_path, recursive=True)
        for file_path in files:
            vllm_label, stt_label = None, None
            for part in file_path.split(os.sep):
                v, s = parse_folder_name(part)
                if v and s:
                    vllm_label, stt_label = v, s
                    break
            if not vllm_label: continue
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
                for m in POWER_METRICS:
                    if m in data:
                        record[m] = data[m]
                        has_data = True
                if has_data: data_records.append(record)
            except Exception as e:
                pass
    return pd.DataFrame(data_records)

def load_latency_data(base_dirs, file_pattern):
    data_records = []
    if isinstance(base_dirs, str): base_dirs = [base_dirs]
    print(f"\n--- Scanning for Latency Data ({file_pattern}) ---")

    for base_dir in base_dirs:
        search_path = os.path.join(base_dir, "**", file_pattern)
        files = glob.glob(search_path, recursive=True)
        for file_path in files:
            vllm_label, stt_label = None, None
            for part in file_path.split(os.sep):
                v, s = parse_folder_name(part)
                if v and s:
                    vllm_label, stt_label = v, s
                    break
            if not vllm_label: continue
            qps_val = parse_qps_from_path(file_path)
            try:
                df = pd.read_csv(file_path)
                if 'end_to_end_end' in df.columns and 'end_to_end_start' in df.columns:
                    latencies = df['end_to_end_end'] - df['end_to_end_start']
                    stats = {
                        "qps": qps_val,
                        "vllm_label": vllm_label,
                        "stt_label": stt_label,
                        "vllm_freq": FREQ_MAP.get(vllm_label, 0),
                        "stt_freq": FREQ_MAP.get(stt_label, 0),
                        "source_file": file_path
                    }
                    for p in LATENCY_PERCENTILES:
                        stats[f"p{p}"] = latencies.quantile(p / 100.0)
                    data_records.append(stats)
            except Exception as e:
                pass
    return pd.DataFrame(data_records)

def generate_best_config_table(df, output_dir, optimize_metric, prefix):
    if df.empty or optimize_metric not in df.columns: return

    group_cols = ["qps", "vllm_freq", "stt_freq", "vllm_label", "stt_label"]
    df_agg = df.groupby(group_cols)[optimize_metric].agg(['mean', 'std']).reset_index()

    best_indices = df_agg.groupby("qps")['mean'].idxmin()
    best_df = df_agg.loc[best_indices].sort_values(by="qps")
    
    # --- NEW: Safe formatting to remove NaN if std dev isn't possible ---
    def format_result(row):
        if pd.isna(row['std']) or row['std'] == 0:
            return f"{row['mean']:.4f}"
        return f"{row['mean']:.4f} ± {row['std']:.4f}"

    best_df['result'] = best_df.apply(format_result, axis=1)
    
    final_table = best_df[["qps", "vllm_freq", "stt_freq", "vllm_label", "stt_label", "result"]]
    metric_title = TITLES.get(optimize_metric, optimize_metric)
    final_table = final_table.rename(columns={"result": f"{metric_title} (Mean ± Std)"})

    print(f"\n=== Best {prefix} Configurations per QPS (Optimizing {optimize_metric}) ===")
    print(final_table.to_string(index=False))
    
    csv_path = os.path.join(output_dir, f"best_configs_{prefix}_{optimize_metric}.csv")
    final_table.to_csv(csv_path, index=False)


def plot_combined_summary_landscape(df_lat, df_pow, output_dir, lat_metric="p99", pow_metric="ENERGY", qps_list=None):
    if df_lat.empty or df_pow.empty:
        print("Missing Data for combined summary.")
        return

    # If no list passed, extract dynamically
    if not qps_list:
        qps_list = sorted(list(set(df_lat['qps'].unique()) & set(df_pow['qps'].unique())))

    df_lat = df_lat[df_lat['qps'].isin(qps_list)]
    df_pow = df_pow[df_pow['qps'].isin(qps_list)]
    if df_lat.empty or df_pow.empty: return

    print(f"\nGenerating Landscape Combined Summary for QPS: {qps_list}")

    lat_agg = df_lat.groupby(["qps", "vllm_freq", "stt_freq"])[lat_metric].mean()
    pow_agg = df_pow.groupby(["qps", "vllm_freq", "stt_freq"])[pow_metric].mean()
    lat_vmin, lat_vmax = lat_agg.min(), lat_agg.max()
    pow_vmin, pow_vmax = pow_agg.min(), pow_agg.max()

    num_cols = len(qps_list)
    # Dynamically scale width based on number of columns
    fig_width = max(10, 6.5 * num_cols + 2)
    
    # squeeze=False guarantees axes is always a 2D array, even if there's only 1 QPS
    fig, axes = plt.subplots(2, num_cols, figsize=(fig_width, 10), squeeze=False)
    plt.subplots_adjust(right=0.9, wspace=0.15, hspace=0.3)
    
    sorted_qps = sorted(qps_list)

    row_configs = [
        {"df": df_lat, "metric": lat_metric, "cmap": "YlOrRd", "vmin": lat_vmin, "vmax": lat_vmax, "row_idx": 0, "label": "Latency (s)", "cbar_pos": [0.92, 0.53, 0.015, 0.35]},
        {"df": df_pow, "metric": pow_metric, "cmap": "viridis", "vmin": pow_vmin, "vmax": pow_vmax, "row_idx": 1, "label": "Energy (kWh)", "cbar_pos": [0.92, 0.12, 0.015, 0.35]}
    ]

    for config in row_configs:
        row_idx = config["row_idx"]
        metric = config["metric"]
        df = config["df"]
        cmap = config["cmap"]
        vmin, vmax = config["vmin"], config["vmax"]

        for col_idx, qps in enumerate(sorted_qps):
            ax = axes[row_idx, col_idx]
            sub_df = df[df['qps'] == qps]

            if not sub_df.empty:
                agg = sub_df.groupby(["vllm_freq", "stt_freq"])[metric].agg(['mean', 'std']).reset_index()
                
                pivot_mean = agg.pivot(index="vllm_freq", columns="stt_freq", values="mean")
                pivot_std = agg.pivot(index="vllm_freq", columns="stt_freq", values="std").fillna(0)
                
                pivot_mean.sort_index(ascending=False, inplace=True)
                pivot_mean.sort_index(axis=1, ascending=True, inplace=True)
                pivot_std.sort_index(ascending=False, inplace=True)
                pivot_std.sort_index(axis=1, ascending=True, inplace=True)

                # --- NEW: Safe format for the Heatmap Text to drop NaN ---
                annot_labels = pd.DataFrame(index=pivot_mean.index, columns=pivot_mean.columns)
                for col in pivot_mean.columns:
                    for idx in pivot_mean.index:
                        m = pivot_mean.at[idx, col]
                        s = pivot_std.at[idx, col]
                        if pd.isna(s) or s == 0:
                            annot_labels.at[idx, col] = f"{m:.2f}"
                        else:
                            annot_labels.at[idx, col] = f"{m:.2f}\n±{s:.2f}"

                sns.heatmap(pivot_mean, ax=ax, cmap=cmap, annot=annot_labels, fmt="", 
                            vmin=vmin, vmax=vmax, cbar=False, 
                            annot_kws={"size": 14})

                ax.set_title(f"{qps} QPS - {TITLES.get(metric, metric)}", fontsize=14)
                
                if col_idx > 0: ax.set_ylabel("") 
                else: ax.set_ylabel("MM LLM GPU Frequency (MHz)", fontsize=13)
                    
                if row_idx == 0: ax.set_xlabel("") 
                else: ax.set_xlabel("STT GPU Frequency (MHz)", fontsize=13)
                
                ax.tick_params(axis='both', which='major', labelsize=12)

            else:
                 ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=12)

        cbar_ax = fig.add_axes(config["cbar_pos"])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) 
        fig.colorbar(sm, cax=cbar_ax, label=config["label"])
        cbar_ax.yaxis.label.set_size(13)

    out_file = os.path.join(output_dir, "combined_summary_landscape_dynamic.pdf")
    plt.savefig(out_file, bbox_inches="tight") 
    plt.close(fig)
    print(f"Saved Landscape PDF (With Std Dev if applicable): {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", nargs='+', type=str, help="Experiment directories")
    parser.add_argument("--power_file", type=str, default="*_summary.json")
    parser.add_argument("--latency_file", type=str, default="frames-*.csv")
    parser.add_argument("--output", type=str, default=".")
    
    parser.add_argument("--combined_summary", action="store_true", help="Generate landscape grid")
    parser.add_argument("--qps_list", nargs='+', type=float, help="List of QPS values to include in the combined summary")
    parser.add_argument("--opt_latency", type=str, default="p99", choices=LATENCY_METRICS)
    parser.add_argument("--opt_power", type=str, default="ENERGY", choices=POWER_METRICS)

    args = parser.parse_args()
    if not os.path.exists(args.output): os.makedirs(args.output)

    df_power = load_power_data(args.input_dirs, args.power_file)
    df_latency = load_latency_data(args.input_dirs, args.latency_file)

    if not df_power.empty:
        generate_best_config_table(df_power, args.output, args.opt_power, "Power")

    if not df_latency.empty:
        generate_best_config_table(df_latency, args.output, args.opt_latency, "Latency")

    if args.combined_summary:
        plot_combined_summary_landscape(
            df_latency, 
            df_power, 
            args.output, 
            lat_metric=args.opt_latency, 
            pow_metric=args.opt_power,
            qps_list=args.qps_list
        )