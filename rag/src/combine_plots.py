"""
Timeline subplot plot for GPU SMACT, CPU utilization, Memory usage, and request counts.

Usage:
    python plot_timeline.py -m path_to_metrics_dir -t path_to_request_timeline.csv
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

width = 5
height = width * 1.2

plt.rcParams.update(
    {
        "figure.figsize": (width, height),
        "font.size": 10,  # Base font size
        "axes.labelsize": 10,  # Size of x and y labels
        "xtick.labelsize": 8,  # Size of tick labels
        "ytick.labelsize": 8,
        "legend.fontsize": 8,  # Size of legend text
        "savefig.dpi": 600,  # High resolution for print
        "figure.autolayout": True,  # Similar to tight_layout()
    }
)


def find_file(directory, prefix, suffix=".csv"):
    pattern = os.path.join(directory, f"{prefix}*{suffix}")
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {pattern}, found {len(matches)}"
        )
    return matches[0]


def load_gpu(path):
    """Returns a Series indexed by absolute epoch seconds (integer), values = sum(SMACT)."""
    df = pd.read_csv(path)
    df_summed = df.groupby("timestamp")["SMACT"].sum().reset_index()
    df_summed["t"] = pd.to_datetime(df_summed["timestamp"]).astype("int64") // 10**6
    return df_summed.groupby("t")["SMACT"].mean().sort_index()


def load_cpu(path):
    """Returns a Series indexed by absolute epoch seconds (integer), values = sum(total_cpu_utilization)."""
    df = pd.read_csv(path)
    df_summed = df.groupby("time")["total_cpu_utilization"].sum().reset_index()
    df_summed["t"] = pd.to_datetime(df_summed["time"]).astype("int64") // 10**6
    return df_summed.groupby("t")["total_cpu_utilization"].mean().sort_index()


def load_mem(path):
    """Returns a Series indexed by absolute epoch seconds (integer), values = mean(%memused)."""
    df = pd.read_csv(path)
    df_summed = df.groupby("time")["%memused"].sum().reset_index()
    df_summed["t"] = pd.to_datetime(df_summed["time"]).astype("int64") // 10**6
    return df_summed.groupby("t")["%memused"].mean().sort_index()


def load_requests(path):
    """Returns a DataFrame indexed by absolute epoch seconds (integer).

    Uses an event-based approach: place +1 at the start second and -1 at the
    end second for each request, then cumsum gives active count per second.
    """
    df = pd.read_csv(path)

    phase_cols = {
        "retrieve": ("retrieve_0_start", "retrieve_0_end"),
        "generate": ("generate_0_start", "generate_0_end"),
    }

    for s_col, e_col in phase_cols.values():
        for col in (s_col, e_col):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    all_times = pd.concat(
        [
            df[col].dropna()
            for s_col, e_col in phase_cols.values()
            for col in (s_col, e_col)
            if col in df.columns
        ]
    )
    t_min = int(np.floor(all_times.min()))
    t_max = int(np.ceil(all_times.max()))
    seconds = np.arange(t_min, t_max + 1)

    result = {}
    for phase, (s_col, e_col) in phase_cols.items():
        events = pd.Series(0, index=seconds, dtype=int)
        if s_col in df.columns and e_col in df.columns:
            for _, row in df[[s_col, e_col]].dropna().iterrows():
                s = int(np.floor(row[s_col]))
                e = int(np.ceil(row[e_col]))
                if t_min <= s <= t_max:
                    events[s] += 1
                if t_min <= e <= t_max:
                    events[e] -= 1
        result[f"active_{phase}"] = events.cumsum()

    return pd.DataFrame(result, index=seconds)


def bookend(series, global_min, global_max, fill_value=0):
    """Add zero-value points at global_min and global_max if not already present.

    This is far cheaper than reindexing over a dense range — we only add at
    most two extra points, and matplotlib draws a flat line between them and
    the nearest real data automatically.
    """
    pts = {}
    if global_min not in series.index:
        pts[global_min] = fill_value
    if global_max not in series.index:
        pts[global_max] = fill_value
    if pts:
        series = pd.concat([series, pd.Series(pts)]).sort_index()
    return series


def main():
    parser = argparse.ArgumentParser(
        description="Plot GPU, CPU, Memory, and request timeline subplots"
    )
    parser.add_argument(
        "-m",
        "--metrics_dir",
        required=True,
        help="Directory containing dcgmi-local-*.csv, sar-cpu-*.csv, sar-mem-*.csv",
    )
    parser.add_argument(
        "-t",
        "--timeline",
        required=True,
        help="Path to request timeline CSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="timeline_subplots.pdf",
        help="Path to output file",
    )
    args = parser.parse_args()

    gpu_path = find_file(args.metrics_dir, "dcgmi-local-")
    cpu_path = find_file(args.metrics_dir, "sar-cpu-")
    mem_path = find_file(args.metrics_dir, "sar-mem-")

    print(f"GPU file:    {gpu_path}")
    print(f"CPU file:    {cpu_path}")
    print(f"Memory file: {mem_path}")

    print("Loading GPU data...")
    gpu_s = load_gpu(gpu_path)

    print("Loading CPU data...")
    cpu_s = load_cpu(cpu_path)

    print("Loading memory data...")
    mem_s = load_mem(mem_path)

    print("Loading request timeline data...")
    req_df = load_requests(args.timeline)
    print("Done loading.")

    # ── Determine the full shared time range (absolute epoch seconds) ──────
    global_min = min(
        gpu_s.index.min(), cpu_s.index.min(), mem_s.index.min(), req_df.index.min()
    )
    global_max = max(
        gpu_s.index.max(), cpu_s.index.max(), mem_s.index.max(), req_df.index.max()
    )
    t0 = global_min

    # Bookend each series with zeros at the global min/max boundaries.
    # This anchors all lines to the same x-range without an expensive reindex.
    gpu_vals = bookend(gpu_s * 100, global_min, global_max)
    cpu_vals = bookend(cpu_s, global_min, global_max)
    mem_vals = bookend(mem_s, global_min, global_max)
    req_retr = bookend(req_df["active_retrieve"], global_min, global_max)
    req_gen = bookend(req_df["active_generate"], global_min, global_max)

    # Shift all indices so x-axis shows seconds-since-start
    def plot_xy(s):
        return s.index - t0, s.values

    # ── Subplots (shared x-axis) ───────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, sharex=True)

    # --- GPU SMACT ---
    ax = axes[0]
    ax.plot(*plot_xy(gpu_vals), color="#1f77b4", linewidth=1.5)
    ax.set_ylabel("GPU SM Active (%)")
    ax.grid(True, alpha=0.3)

    # --- CPU utilization ---
    ax = axes[1]
    ax.plot(*plot_xy(cpu_vals), color="#ff7f0e", linewidth=1.5)
    ax.set_ylabel("CPU Util. (%)")
    ax.grid(True, alpha=0.3)

    # --- Memory ---
    ax = axes[2]
    ax.plot(*plot_xy(mem_vals), color="#9467bd", linewidth=1.5)
    ax.set_ylabel("DRAM Usage (%)")
    ax.grid(True, alpha=0.3)

    # --- Active retrieve / generate (steps-post for vertical jumps) ---
    ax = axes[3]
    ax.plot(
        *plot_xy(req_retr),
        color="#2ca02c",
        linewidth=2,
        linestyle="-",
        drawstyle="steps-post",
        label="Retrieve",
    )
    ax.plot(
        *plot_xy(req_gen),
        color="#d62728",
        linewidth=2,
        linestyle=":",
        drawstyle="steps-post",
        label="Generate",
    )
    ax.set_ylabel("Active Reqs.")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=12)

    fig.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
