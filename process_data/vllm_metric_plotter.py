"""
vLLM cache metrics visualization utilities.

Plots prefix cache hit rates, KV cache usage, and block lifetime metrics
from vLLM monitoring CSV logs.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_log(log_file, output_file):
    log_path = Path(log_file)
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_path}'", file=sys.stderr)
        print("Did your main script run correctly?", file=sys.stderr)
        return
    except pd.errors.EmptyDataError:
        print(
            f"Error: Log file '{log_path}' is empty. No data to plot.", file=sys.stderr
        )
        return

    for metric in [
        "running_avg_hit_rate",
        "interval_hit_rate",
        "kv_block_lifetime_avg",
        "kv_block_idle_before_evict_avg",
        "kv_block_reuse_gap_avg",
    ]:
        if metric not in df.columns or df.empty:
            print(
                f"Error: CSV file is missing data or required columns: {metric}",
                file=sys.stderr,
            )
            return

    # Calculate percentage columns for hit rates
    df["running_avg_hit_rate_pct"] = df["running_avg_hit_rate"] * 100
    df["interval_hit_rate_pct"] = df["interval_hit_rate"] * 100

    # --- Print Final Summary ---
    final_stats = df.iloc[-1]
    print("\n--- Final Job Summary ---")
    print(
        f"Total Runtime (approx):                 {final_stats['elapsed_sec']:.0f} seconds"
    )
    print(
        f"Total Cache Queries:                    {final_stats['job_total_queries']:,.0f}"
    )
    print(
        f"Total Cache Hits:                       {final_stats['job_total_hits']:,.0f}"
    )
    print(
        f"Final KV Cache Usage:                   {final_stats['kv_cache_usage_percent']:.4f}%"
    )
    print(
        f"Final KV Block Lifetime Avg.:           {final_stats['kv_block_lifetime_avg']:.4f}"
    )
    print(
        f"Final KV Block Idle Before Evict Avg.:  {final_stats['kv_block_idle_before_evict_avg']:.4f}"
    )
    print(
        f"Final KV Block Reuse Gap Avg.:          {final_stats['kv_block_reuse_gap_avg']:.4f}"
    )
    print(
        f"**Final Avg. Hit Rate:                  {final_stats['running_avg_hit_rate']:.2%}**"
    )

    cumulative_metrics = [
        "job_total_hits",
        "job_total_queries",
        "kv_block_lifetime_avg",
        "kv_block_idle_before_evict_avg",
        "kv_block_reuse_gap_avg",
    ]
    labels = [
        "Total Hits",
        "Total Queries",
        "KV Block Lifetime Average",
        "KV Block Idle Before Evict Average",
        "KV Block Reuse Gap Average",
    ]
    colors = [
        "#377eb8",
        "#e41a1c",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
    ]
    units = [
        "Counts (Cached Tokens)",
        "Counts (Queried Tokens)",
        "Seconds",
        "Seconds",
        "Seconds",
    ]
    fig_height = 4 * (len(cumulative_metrics) + 1) + 4
    num_plots = len(cumulative_metrics) + 1

    fig, axes = plt.subplots(num_plots, 1, figsize=(14, fig_height), sharex=True)
    fig.suptitle("vLLM Prefix Cache & GPU Usage Analysis", fontsize=16, y=1.02)

    for index in range(len(cumulative_metrics)):
        ax = axes[index]
        ax.plot(
            df["elapsed_sec"],
            df[cumulative_metrics[index]],
            label=labels[index],
            color=colors[index],
            linewidth=2,
        )
        ax.set_title(labels[index])
        ax.set_ylabel(units[index])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left")

    ax = axes[-1]
    ax.plot(
        df["elapsed_sec"],
        df["kv_cache_usage_percent"],
        label="KV Cache Usage",
        color="#a65628",
        linewidth=2,
    )
    ax.set_title("KV Cache Usage")
    ax.set_ylabel("Usage (%)")
    ax.set_xlabel("Time (seconds)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0, 105)  # Fix Y-axis to 0-100%
    ax.legend(loc="upper left")

    plt.tight_layout()
    # Save the plot
    plt.savefig(output_file)
    print(f"\nPlot saved to '{output_file}'")
    plt.close(fig)


def find_vllm_csv_files(vllm_dir: Path) -> list:
    vllm_pattern = re.compile(r"^vllm.*\.csv$")
    matches = [
        str(p.relative_to(vllm_dir))
        for p in vllm_dir.rglob("*")
        if vllm_pattern.match(p.name)
    ]
    return sorted(matches)


def process_and_plot_vllm_metric(raw_dir: Path, processed_dir: Path, metric_dir: Path):
    vllm_files = find_vllm_csv_files(raw_dir)
    for vllm_file in vllm_files:
        shutil.copy(raw_dir / vllm_file, processed_dir / vllm_file)
        output_file = metric_dir / vllm_file.split("/")[0] / f"vllm_cache_plot.pdf"
        plot_log(processed_dir / vllm_file, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot vLLM monitor logs.")
    parser.add_argument(
        "log_file",
        help="Input CSV log file (e.g., path/to/output_dir/vllm_cache_log.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="vllm_cache_plot.pdf",
        help="Path to the vllm plot pdf file (default: vllm_cache_plot.pdf)",
    )
    args = parser.parse_args()

    plot_log(args.log_file, args.output)
