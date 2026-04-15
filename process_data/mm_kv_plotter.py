import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_kv_metrics(df, output_path):
    """Generates the KV (Prefix) Cache & Usage Plot."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    ax1, ax2, ax3 = axes

    fig.suptitle("vLLM KV (Text) Cache & Memory Analysis", fontsize=16, y=0.95)

    ax1.plot(df["elapsed_sec"], df["job_total_queries"], label="Total Queries", color="green", linewidth=2, alpha=0.7)
    ax1.plot(df["elapsed_sec"], df["job_total_hits"], label="Total Hits", color="blue", linewidth=2)
    ax1.set_title("Prefix Cache Volume")
    ax1.set_ylabel("Token Count")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")


    ax2.plot(df["elapsed_sec"], df["prefix_hit_rate_pct"], label="Running Avg Hit Rate", color="blue", linewidth=2)
    ax2.set_title("Prefix Cache Efficiency")
    ax2.set_ylabel("Hit Rate (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    ax3.plot(df["elapsed_sec"], df["kv_cache_usage_percent"], label="KV Block Usage", color="tab:red", linewidth=2)
    ax3.set_title("GPU KV Cache Memory Saturation")
    ax3.set_ylabel("Usage (%)")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylim(-5, 105)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper left")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"Saved KV plot to: {output_path}")
    plt.close(fig)


def plot_mm_metrics(df, output_path):
    """Generates the Multi-Modal Cache Plot."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1, ax2 = axes

    fig.suptitle("vLLM Multi-Modal (Image/Video) Cache Analysis", fontsize=16, y=0.95)

    ax1.plot(df["elapsed_sec"], df["mm_total_queries"], label="MM Queries", color="purple", linewidth=2, alpha=0.7)
    ax1.plot(df["elapsed_sec"], df["mm_total_hits"], label="MM Hits", color="orange", linewidth=2)
    ax1.set_title("Multi-Modal Cache Volume")
    ax1.set_ylabel("Item Count")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2.plot(df["elapsed_sec"], df["mm_hit_rate_pct"], label="MM Hit Rate", color="orange", linewidth=2)
    ax2.set_title("Multi-Modal Cache Efficiency")
    ax2.set_ylabel("Hit Rate (%)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"Saved MM plot to: {output_path}")
    plt.close(fig)


def process_log(log_file, output_base, output_dir=None):
    log_path = Path(log_file)
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        return

    if "mm_running_avg_hit_rate" not in df.columns:
        df["mm_running_avg_hit_rate"] = 0.0
        df["mm_total_hits"] = 0
        df["mm_total_queries"] = 0

    df["prefix_hit_rate_pct"] = df["running_avg_hit_rate"] * 100
    df["mm_hit_rate_pct"] = df["mm_running_avg_hit_rate"] * 100

    final = df.iloc[-1]
    print(f"\n--- Summary for {log_path.name} ---")
    print(f"Runtime: {final['elapsed_sec']:.0f}s")
    print(f"KV Final Usage: {final['kv_cache_usage_percent']:.1f}%")
    print(f"Prefix Hit Rate: {final['running_avg_hit_rate']:.2%}")
    print(f"MM Hit Rate:     {final['mm_running_avg_hit_rate']:.2%}")
    print("-----------------------------------")

    out_path = Path(output_base)

    if output_dir:
        out_dir_path = Path(output_dir)
        # Create directory if it doesn't exist (mkdir -p)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        # Combine directory with the filename part of output_base
        out_path = out_dir_path / out_path.name
    
    base_name = out_path.stem
    extension = out_path.suffix if out_path.suffix else ".pdf"
    parent = out_path.parent

    kv_file = parent / f"{base_name}_kv{extension}"
    mm_file = parent / f"{base_name}_mm{extension}"

    # Create Plots
    plot_kv_metrics(df, kv_file)
    plot_mm_metrics(df, mm_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot vLLM monitor logs.")
    parser.add_argument(
        "log_file",
        help="Input CSV log file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="vllm_plot.pdf",
        help="Base filename for outputs (default: vllm_plot.pdf)",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        default=None,
        help="Directory to save the plots. Will be created if it does not exist.",
    )
    args = parser.parse_args()

    process_log(args.log_file, args.output, args.output_dir)