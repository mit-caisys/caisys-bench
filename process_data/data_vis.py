"""
Data visualization utilities for performance analysis.

Provides functions to plot CDFs, timelines, accuracies, and latency metrics
from experimental run data stored in CSV files.
"""

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def plot_cdf(
    df,
    output_folder,
    node_name,
    run_name,
    smooth=False,
    ax=None,
    latency=False,
    color=None,
):
    """
    Plot cumulative distribution function (CDF) for a metric.

    Args:
        df: DataFrame containing the data.
        output_folder: Directory to save the plot.
        node_name: Column name or prefix for the metric.
        run_name: Name for this run (used in labels and filename).
        smooth: Whether to overlay a KDE-smoothed curve.
        ax: Optional axes to plot on (for combined plots).
        latency: If True, calculate latency as end - start time.
        color: Optional color for the plot line.
    """
    if latency:
        start_col = f"{node_name}_start"
        end_col = f"{node_name}_end"

        if start_col not in df.columns or end_col not in df.columns:
            return

        df_valid = df.dropna(subset=[start_col, end_col])
        if df_valid.empty:
            return

        is_individual_plot = ax is None
        if is_individual_plot:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.gca()

        data = df_valid[end_col] - df_valid[start_col]
    else:
        is_individual_plot = ax is None
        if is_individual_plot:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.gca()

        data = df[node_name]

    pretty_label = re.sub(r"[-_]", " ", run_name)
    x_ecdf = np.sort(data)
    y_ecdf = np.arange(1, len(x_ecdf) + 1) / len(x_ecdf)
    ax.plot(
        x_ecdf, y_ecdf, marker=".", linestyle="none", label=pretty_label, color=color
    )

    if smooth:
        kde = sm.nonparametric.KDEUnivariate(data)
        kde.fit()
        ax.plot(kde.support, kde.cdf, lw=2, color=color)

    if is_individual_plot:
        pretty_name = node_name.replace("_", " ").title()
        if latency:
            ax.set_title(f"CDF of {pretty_name} Latency for {run_name}")
            ax.set_xlabel(f"{pretty_name} Latency (seconds)")
            ax.set_ylabel("Cumulative Probability")
        else:
            ax.set_title(f"CDF of {pretty_name} for {run_name}")
            ax.set_xlabel(f"{pretty_name}")
            ax.set_ylabel("Cumulative Probability")
        ax.legend()
        ax.grid(True)
        indv_folder = os.path.join(output_folder, f"{run_name}")
        os.makedirs(indv_folder, exist_ok=True)

        plt.savefig(os.path.join(indv_folder, f"cdf_{run_name}_{node_name}.pdf"))
        plt.close()


def analyze_runs_cdf(
    input_folder,
    output_folder,
    file_format,
    node_to_analyze,
    latency=False,
    sorting_lambda=None,
):
    """
    Finds all matching CSVs in an input folder and creates plots in an output folder,
    skipping files with missing data.
    """
    search_path = os.path.join(input_folder, file_format)
    csv_files_found = glob.glob(search_path)

    csv_files = sorted(csv_files_found, key=sorting_lambda)

    if not csv_files:
        print(f"No CSV files found in '{input_folder}'.")
        return

    fig, combined_ax = plt.subplots(figsize=(12, 7))

    num_files = len(csv_files)

    colors = plt.cm.viridis(np.linspace(0, 1, num_files))

    for i, file_path in enumerate(csv_files):
        run_name = os.path.basename(file_path).replace(".csv", "")
        df = pd.read_csv(file_path)

        color = colors[i]

        plot_cdf(
            df,
            output_folder,
            node_to_analyze,
            run_name,
            smooth=True,
            ax=combined_ax,
            latency=latency,
            color=color,
        )
        plot_cdf(
            df,
            output_folder,
            node_to_analyze,
            run_name,
            smooth=True,
            latency=latency,
            color=color,
        )

    pretty_name = re.sub(r"[-_]", " ", node_to_analyze)
    if latency:
        combined_ax.set_title(f"Combined CDF of {pretty_name} Latency")
        combined_ax.set_xlabel(f"{pretty_name} Latency (seconds)")
        combined_ax.set_ylabel("Cumulative Probability")
    else:
        combined_ax.set_title(f"Combined CDF of {pretty_name}")
        combined_ax.set_xlabel(f"{pretty_name}")
        combined_ax.set_ylabel("Cumulative Probability")
    combined_ax.legend()
    combined_ax.grid(True)
    plt.savefig(os.path.join(output_folder, f"combined_cdf_{node_to_analyze}.pdf"))
    plt.close()


def plot_accuracies(input_folder, output_folder, file_format: str, sorting_lamba=None):
    """
    Plots bar graph of accuracies given an input folder of csv files
    """

    search_path = os.path.join(input_folder, file_format)
    csv_files_found = glob.glob(search_path)

    csv_files = sorted(csv_files_found, key=sorting_lamba)

    accuracies = {}

    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)

            if "correct" in df.columns:
                accuracy = df["correct"].mean()

                filename = os.path.basename(filepath)
                filename_no_ext = os.path.splitext(filename)[0]
                accuracies[filename_no_ext] = accuracy
            else:
                print(
                    f"  - Warning: 'correct' column not found in '{os.path.basename(filepath)}'. Skipping."
                )

        except Exception as e:
            print(f"Error processing file {os.path.basename(filepath)}: {e}")

    if not accuracies:
        print("Could not calculate any accuracies. Please check your CSV files.")
        return

    labels = accuracies.keys()
    values = list(accuracies.values())

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create bars
    bars = ax.bar(labels, values, color=plt.cm.viridis(np.array(values) / max(values)))

    # Add labels and title
    ax.set_xlabel("Configurations", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Rotate x-axis labels for better readability if they are long
    plt.xticks(rotation=45, ha="right", fontsize=12)

    # Add the accuracy value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.02,
            f"{yval:.2%}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Add a grid for better readability
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust layout to make sure everything fits
    plt.tight_layout()

    # Display the plot
    plt.savefig(os.path.join(output_folder, f"accuracies.pdf"))
    plt.close()


def plot_timeline(df, output_folder, nodes, run_name):

    df.replace("", np.nan, inplace=True)
    fig, ax = plt.subplots(figsize=(12, 8))

    # 1. Find the global start time for the entire experiment.
    # This will be our reference point (time = 0).
    start_cols = [f"{n}_start" for n in nodes if f"{n}_start" in df.columns]
    end_cols = [f"{n}_end" for n in nodes if f"{n}_end" in df.columns]
    if not start_cols:
        print("Warning: No start columns found in DataFrame. Cannot generate plot.")
        return

    # Get the minimum timestamp across all start columns, ignoring any missing values
    experiment_start_time = df[start_cols].min().min()
    experiment_end_time = df[end_cols].max().max()

    # If no valid start time is found, exit
    if pd.isna(experiment_start_time):
        print(f"Warning: Could not find a valid start time for run '{run_name}'.")
        return

    for node_prefix in nodes:
        start_col = f"{node_prefix}_start"
        end_col = f"{node_prefix}_end"

        # Check if columns exist and have data
        if (
            start_col in df.columns
            and end_col in df.columns
            and df[start_col].notna().any()
        ):
            # Create a list of (timestamp, event_type) tuples
            events = []
            for index, row in df.dropna(subset=[start_col, end_col]).iterrows():
                events.append((row[start_col], 1))  # 1 for start
                events.append((row[end_col], -1))  # -1 for end

            if not events:
                continue

            events.sort()

            active_requests = 0
            # 2. Keep time points as numeric timestamps, not datetime objects
            time_points_numeric = []
            timeline_values = []

            # This logic creates the "step" plot effect
            for t, event_type in events:
                # Add a point right before the event to create a vertical line
                if time_points_numeric and time_points_numeric[-1] < t:
                    time_points_numeric.append(t)
                    timeline_values.append(
                        timeline_values[-1]
                    )  # Keep previous request count

                active_requests += event_type
                time_points_numeric.append(t)
                timeline_values.append(active_requests)

            # 3. Convert absolute timestamps to time elapsed since the global start
            relative_time_points = [
                t - experiment_start_time for t in time_points_numeric
            ]
            relative_global_end = experiment_end_time - experiment_start_time

            # Add an initial point at t=0 with 0 requests for a clean start
            if relative_time_points[0] > 0:
                relative_time_points.insert(0, 0)
                timeline_values.insert(0, 0)

            if relative_time_points[-1] < relative_global_end:
                relative_time_points.append(relative_global_end)
                timeline_values.append(0)

            label = node_prefix.replace("_", " ").title()
            ax.step(
                relative_time_points,
                timeline_values,
                where="post",
                marker="o",
                linestyle="-",
                ms=4,
                label=label,
            )

    pretty_name = re.sub(r"[-_]", " ", run_name)
    ax.set_ylabel("Number of Active Requests", fontsize=12)

    # 4. Update axis label and remove date formatting
    ax.set_xlabel("Time Since Start (seconds)", fontsize=12)
    ax.legend(title="Node", fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.xticks(rotation=0, ha="center", fontsize=10)
    plt.yticks(fontsize=10)
    fig.tight_layout()

    indv_folder = os.path.join(output_folder, f"{run_name}")
    os.makedirs(indv_folder, exist_ok=True)

    plt.savefig(os.path.join(indv_folder, f"active_requests_timeline_{run_name}.pdf"))
    plt.close()


def plot_p90_latency(df, output_folder, run_name, line_plot=True):

    indv_folder = os.path.join(output_folder, f"{run_name}")

    df["latency_s"] = df["end_to_end_end"] - df["end_to_end_start"]

    experiment_start_time = df["end_to_end_start"].min()

    # 2. Resample data based on absolute finish times
    df["finish_time"] = pd.to_datetime(df["end_to_end_end"], unit="s")
    df.set_index("finish_time", inplace=True)
    p90_latency = df["latency_s"].resample("30s").quantile(0.9)
    p90_latency = p90_latency.dropna()

    # Exit if there's no data to plot
    if p90_latency.empty:
        print(f"Warning: No data to plot for run '{run_name}' after resampling.")
        return

    # 3. Calculate elapsed time in seconds from the experiment's start
    # We convert the datetime index of our P90 data back to numeric seconds
    # and subtract the experiment's start time.
    time_since_start = (
        p90_latency.index.astype(np.int64) // 10**9
    ) - experiment_start_time

    # --- End of Key Changes ---

    # Get the latency values for the y-axis
    latency_values = p90_latency.values

    # Common plot settings
    plt.figure(figsize=(12, 6))
    plt.ylabel("P90 Latency (s)")
    plt.grid(True, linestyle="--", alpha=0.7)

    # The plotting logic now uses the calculated `time_since_start` for the x-axis
    if not line_plot:
        plt.bar(time_since_start, latency_values, width=25)
        plt.xlabel("Time Since Start (seconds)")
    else:
        plt.plot(time_since_start, latency_values, marker="o", linestyle="-")
        plt.xlabel("Time Since Start (seconds)")

    plt.tight_layout()

    plt.savefig(os.path.join(indv_folder, f"p90_latency_line_plot_{run_name}.pdf"))
    plt.close()


def plot_timelines_and_p90(
    input_folder,
    output_folder,
    file_format,
    nodes,
    plot_names: list[str] = ["timeline", "p90_latency"],
):
    search_path = os.path.join(input_folder, file_format)
    csv_files_found = glob.glob(search_path)

    csv_files = sorted(csv_files_found)

    if not csv_files:
        print(f"No CSV files found in '{input_folder}'.")
        return

    for file_path in csv_files:
        run_name = os.path.basename(file_path).replace(".csv", "")
        df = pd.read_csv(file_path)

        if "timeline" in plot_names:
            plot_timeline(df, output_folder, nodes, run_name)

        if "p90_latency" in plot_names:
            plot_p90_latency(df, output_folder, run_name)


# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and plot performance data from CSV files."
    )

    # --- Positional Arguments ---
    parser.add_argument(
        "output_folder", help="The directory where output plots will be saved."
    )
    parser.add_argument(
        "input_folder", help="The directory containing the input CSV files."
    )

    # --- Optional Arguments for Node Lists and File Format ---
    parser.add_argument(
        "--file-format",
        default="*.csv",
        help="Pattern to match input files (e.g., '*.csv', 'run-*.csv'). Default is '*.csv'.",
    )
    parser.add_argument(
        "--latency-nodes",
        nargs="+",
        help="One or more node names for which to plot latency CDFs (e.g., retrieve_0 generate_0).",
    )
    parser.add_argument(
        "--value-nodes",
        nargs="+",
        help="One or more column names for which to plot value CDFs (e.g., total_input_tokens).",
    )
    parser.add_argument(
        "--timeline-nodes",
        nargs="+",
        help="One or more node names for which to plot active request timelines.",
    )

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_folder, exist_ok=True)

    if args.latency_nodes:
        print("Analyzing latency CDFs for:", args.latency_nodes)
        for node in args.latency_nodes:
            analyze_runs_cdf(
                args.input_folder,
                args.output_folder,
                args.file_format,  # Replaced "*.csv"
                node,
                latency=True,
            )

    # Plot value CDFs if value-nodes are specified
    if args.value_nodes:
        print("Analyzing value CDFs for:", args.value_nodes)
        for node in args.value_nodes:
            analyze_runs_cdf(
                args.input_folder,
                args.output_folder,
                args.file_format,
                node,
            )

    # Plot timelines and P90 latency if timeline-nodes are specified
    if args.timeline_nodes:
        print("Analyzing timelines for:", args.timeline_nodes)
        plot_timelines_and_p90(
            args.input_folder, args.output_folder, args.file_format, args.timeline_nodes
        )

    # Always plot accuracies
    print("Plotting accuracies...")
    plot_accuracies(args.input_folder, args.output_folder, args.file_format)

    print("Analysis complete. Plots are saved in:", args.output_folder)
