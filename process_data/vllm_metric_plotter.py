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

    if "running_avg_hit_rate" not in df.columns or df.empty:
        print("Error: CSV file is missing data or required columns.", file=sys.stderr)
        return

    # Calculate percentage columns for hit rates
    df["running_avg_hit_rate_pct"] = df["running_avg_hit_rate"] * 100
    df["interval_hit_rate_pct"] = df["interval_hit_rate"] * 100

    # --- Print Final Summary ---
    final_stats = df.iloc[-1]
    print("\n--- Final Job Summary ---")
    print(f"Total Runtime (approx): {final_stats['elapsed_sec']:.0f} seconds")
    print(f"Total Cache Queries:    {final_stats['job_total_queries']:,.0f}")
    print(f"Total Cache Hits:       {final_stats['job_total_hits']:,.0f}")
    print(f"Final KV Cache Usage:   {final_stats['kv_cache_usage_percent']:.1f}%")
    print(f"**Final Avg. Hit Rate:  {final_stats['running_avg_hit_rate']:.2%}**")

    fig_height = 14
    num_plots = 3

    fig, axes = plt.subplots(num_plots, 1, figsize=(14, fig_height), sharex=True)
    ax1, ax2, ax3 = axes

    fig.suptitle("vLLM Prefix Cache & GPU Usage Analysis", fontsize=16, y=1.02)

    ax1.plot(
        df["elapsed_sec"],
        df["job_total_hits"],
        label="Total Hits",
        color="b",
        linewidth=2,
    )
    ax1.set_title("Total Hits")
    ax1.set_ylabel("Count (Cached Tokens)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left")

    ax2.plot(
        df["elapsed_sec"],
        df["job_total_queries"],
        label="Total Queries",
        color="g",
        linewidth=2,
    )
    ax2.set_title("Total Queries")
    ax2.set_ylabel("Count (Queried Tokens)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper left")

    ax3.plot(
        df["elapsed_sec"],
        df["kv_cache_usage_percent"],
        label="KV Cache Usage",
        color="tab:red",
        linewidth=2,
    )
    ax3.set_title("KV Cache Usage")
    ax3.set_ylabel("Usage (%)")
    ax3.set_xlabel("Time (seconds)")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.set_ylim(0, 105)  # Fix Y-axis to 0-100%
    ax3.legend(loc="upper left")

    plt.tight_layout()
    # Save the plot
    plt.savefig(output_file)
    print(f"\nPlot saved to '{output_file}'")
    plt.show()


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
