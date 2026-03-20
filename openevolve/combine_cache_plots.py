"""
Cache hit rate plotting utilities.

Combines KV cache metrics from multiple experiments into comparison plots.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

width = 4
height = width / 1.618

plt.rcParams.update(
    {
        "figure.figsize": (width, height),
        "font.size": 12,  # Base font size
        "axes.labelsize": 12,  # Size of x and y labels
        "xtick.labelsize": 10,  # Size of tick labels
        "ytick.labelsize": 10,
        "legend.fontsize": 10,  # Size of legend text
        "savefig.dpi": 600,  # High resolution for print
        "figure.autolayout": True,  # Similar to tight_layout()
    }
)


def plot_hit_rate(csv_paths: list[Path], labels: list[str]):
    """
    Plots job_total_hits / job_total_queries over time for a list of CSVs.
    Time is rescaled to start at 0 for each file.
    """

    for csv_path, label in zip(csv_paths, labels):
        try:
            df = pd.read_csv(csv_path)

            required_columns = ["elapsed_sec", "job_total_hits", "job_total_queries"]
            if not all(col in df.columns for col in required_columns):
                print(
                    f"Skipping {csv_path}: Missing required columns {required_columns}"
                )
                continue

            start_time = df["elapsed_sec"].iloc[0]
            df["time_rescaled"] = df["elapsed_sec"] - start_time

            df["hit_rate"] = df.apply(
                lambda row: (
                    100 * row["job_total_hits"] / row["job_total_queries"]
                    if row["job_total_queries"] > 0
                    else 0.0
                ),
                axis=1,
            )

            plt.plot(df["time_rescaled"], df["hit_rate"], label=label, alpha=0.8)

            print(f"Processed {csv_path}")

        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    plt.xlabel("Time (s)")
    plt.ylabel("KV Cache Hit Rate (%)")
    plt.legend()
    plt.grid(True)

    output_filename = "job_hit_rate_over_time.pdf"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.close()


def plot_lifetime(csv_paths: list[Path], labels: list[str]):
    """
    Plots kv_block_lifetime_avg over time for a list of CSVs.
    Time is rescaled to start at 0 for each file.
    """

    for csv_path, label in zip(csv_paths, labels):
        try:
            df = pd.read_csv(csv_path)

            required_columns = ["elapsed_sec", "kv_block_lifetime_avg"]
            if not all(col in df.columns for col in required_columns):
                print(
                    f"Skipping {csv_path}: Missing required columns {required_columns}"
                )
                continue

            start_time = df["elapsed_sec"].iloc[0]
            df["time_rescaled"] = df["elapsed_sec"] - start_time
            plt.plot(
                df["time_rescaled"], df["kv_block_lifetime_avg"], label=label, alpha=0.8
            )

            print(f"Processed {csv_path}")

        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    plt.xlabel("Time (s)")
    plt.ylabel("KV Block Lifetime (s)")
    plt.legend()
    plt.grid(True)

    output_filename = "kv_block_lifetime_avg_over_time.pdf"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot total hits / total queries over time from CSV logs"
    )
    parser.add_argument(
        "--csv_files", nargs="+", type=Path, help="list of csv files to plot"
    )
    parser.add_argument(
        "--labels", nargs="+", type=str, help="list of corresponding labels"
    )
    args = parser.parse_args()

    plot_hit_rate(args.csv_files, args.labels)
    plot_lifetime(args.csv_files, args.labels)
