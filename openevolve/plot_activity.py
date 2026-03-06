"""
Activity timeline plotting utilities for OpenEvolve.

Generates timeline plots showing active evaluations and LLM requests over time.
"""

import argparse
import csv
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def plot_activity(csv_filepath: Path, output_filepath: Path):
    """
    Reads the activity log CSV, plots active evaluations and LLM requests over time,
    and saves the plot as a PDF.

    Args:
        csv_filepath: Path to the input CSV file.
        output_filepath: Path to the output pdf.
    """
    timestamps = []
    eval_counts = []
    llm_counts = []

    print(f"Reading CSV file: {csv_filepath}")
    try:
        with open(csv_filepath, "r") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header row

            # Check header
            if header != ["timestamp", "active_evaluations", "active_llm_requests"]:
                print(f"Warning: Unexpected CSV header: {header}")
                print(
                    "Expected: ['timestamp', 'active_evaluations', 'active_llm_requests']"
                )
                # Attempt to proceed assuming column order is correct

            first_timestamp = None
            for i, row in enumerate(reader):
                if len(row) != 3:
                    print(
                        f"Warning: Skipping row {i + 2} due to incorrect number of columns: {row}"
                    )
                    continue
                try:
                    # Parse timestamp (handle potential slight variations in format if needed)
                    ts_str = row[0]
                    # Ensure milliseconds are handled correctly (support both .SSS and .ffffff)
                    if "." in ts_str:
                        base, ms = ts_str.split(".")
                        ms = ms.ljust(6, "0")[:6]  # Pad/truncate to microseconds
                        ts = datetime.strptime(f"{base}.{ms}", "%Y-%m-%d %H:%M:%S.%f")
                    else:
                        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")

                    eval_count = int(row[1])
                    llm_count = int(row[2])

                    if first_timestamp is None:
                        first_timestamp = ts

                    timestamps.append(ts)
                    eval_counts.append(eval_count)
                    llm_counts.append(llm_count)

                except ValueError as e:
                    print(
                        f"Warning: Skipping row {i + 2} due to parsing error: {row} - {e}"
                    )
                    continue
                except Exception as e:
                    print(
                        f"Warning: Skipping row {i + 2} due to unexpected error: {row} - {e}"
                    )
                    continue

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    if not timestamps:
        print("No valid data found in the CSV file to plot.")
        return

    # --- Data Preparation for Plotting ---
    # Calculate elapsed time in seconds relative to the first timestamp
    elapsed_seconds = [(ts - first_timestamp).total_seconds() for ts in timestamps]

    # --- Plotting ---
    print("Generating plot...")
    plt.style.use("seaborn-v0_8-darkgrid")  # Use a nice style
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use step plot as counts change instantaneously
    ax.step(
        elapsed_seconds,
        eval_counts,
        where="post",
        label="Active Evaluations",
        linewidth=2,
    )
    ax.step(
        elapsed_seconds,
        llm_counts,
        where="post",
        label="Active LLM Requests",
        linewidth=2,
        linestyle="--",
    )

    # Formatting the plot
    ax.set_xlabel("Time Elapsed (seconds)")
    ax.set_ylabel("Active Count")
    ax.set_title("Active Evaluations and LLM Requests Over Time")
    ax.legend()
    ax.grid(True)  # Ensure grid is visible

    # Improve layout
    plt.tight_layout()

    try:
        plt.savefig(output_filepath, format="pdf", bbox_inches="tight")
        print(f"Plot saved successfully as: {output_filepath}")
    except Exception as e:
        print(f"Error saving plot to PDF: {e}")

    plt.close(fig)


def find_timeline_csv_files(timeline_dir: Path) -> list:
    timeline_pattern = re.compile(r"^request_openevolve.*\.csv$")
    matches = [
        str(p.relative_to(timeline_dir))
        for p in timeline_dir.rglob("*")
        if timeline_pattern.match(p.name)
    ]
    return sorted(matches)


def process_and_plot_activity(processed_dir: Path, metric_dir: Path):
    timeline_files = find_timeline_csv_files(processed_dir)
    for timeline_file in timeline_files:
        output_filepath = (
            metric_dir
            / timeline_file.split("/")[0]
            / f"activity_{timeline_file.split('/')[-1].split('_', 1)[1].split('.')[0]}.pdf"
        )
        plot_activity(processed_dir / timeline_file, output_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot activity log from CSV file and save as PDF."
    )
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument(
        "-o",
        "--output",
        default="timeline_plot.pdf",
        help="Path to the output pdf file (default: timeline_plot.pdf)",
    )

    args = parser.parse_args()
    plot_activity(args.csv_file, args.output)
