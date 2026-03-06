"""
SAR (System Activity Report) data loading and plotting utilities.

Handles parsing of sysstat sar logs for CPU, memory, and network metrics.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

# Spline degree for smoothing plots
DEGREE = 3


def get_date(log_filepath: Path):
    """Extract date from sar log file header (first line)."""
    with open(log_filepath, "r") as f:
        metadata = f.readline()

    # Match format: MM/DD/YYYY
    date_pattern = r"(\d{2}/\d{2}/\d{4})"
    match = re.search(date_pattern, metadata)

    if match:
        raw_date = match.group(1)
        try:
            date = pd.to_datetime(raw_date).strftime("%Y-%m-%d")
        except:
            date = datetime.now().strftime("%Y-%m-%d")
    else:
        date = datetime.now().strftime("%Y-%m-%d")

    return date


def load_sar_cpu_data(log_filepath: Path):
    """
    Loads CPU utilization data from a sar log file.

    Args:
        log_filepath: The path to the sar CPU log file.

    Returns:
        DataFrame with parsed CPU data including calculated total utilization.
    """
    # Skip first 3 header lines and last footer line
    try:
        df = pd.read_csv(
            log_filepath,
            sep=r"\s+",
            skiprows=3,
            skipfooter=1,
            engine="python",
            header=None,
        )
        if df.empty:
            return pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    # Handle 12-hour format (with AM/PM) vs 24-hour format
    if df.iloc[0, 1] in ["AM", "PM"]:
        df.columns = [
            "time_part",
            "am_pm",
            "cpu",
            "user",
            "nice",
            "system",
            "iowait",
            "steal",
            "idle",
        ]
        time_series = df["time_part"] + " " + df["am_pm"]
    else:
        df.columns = [
            "time_part",
            "cpu",
            "user",
            "nice",
            "system",
            "iowait",
            "steal",
            "idle",
        ]
        time_series = df["time_part"]

    # Parse datetime and add date from file header
    date = get_date(log_filepath)
    time_series = time_series.astype(str).apply(
        lambda ts: f"{date} {ts}" if "-" not in ts else ts
    )
    df["time"] = pd.to_datetime(time_series, format="mixed")

    # Filter by pinned CPU cores from directory name (e.g., "logs_0-7")
    cpus_str = log_filepath.parent.name.split("_")[-1]
    pattern = re.compile(r"^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")
    if bool(pattern.match(cpus_str)):
        # Parse CPU list (e.g., "0-3,5,7")
        cpus = set()
        for part in cpus_str.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                cpus.update(range(start, end + 1))
            else:
                cpus.add(int(part))
        df = df[df["cpu"].apply(lambda x: x.isnumeric() and int(x) in cpus)].copy()
    else:
        # Default: include "all" and individual cores
        df = df[(df["cpu"] == "all") | df["cpu"].str.isnumeric()].copy()

    # Calculate total CPU utilization (100% - idle%)
    df["idle"] = pd.to_numeric(df["idle"])
    df["total_cpu_utilization"] = 100 - df["idle"]

    return df[
        [
            "time",
            "cpu",
            "user",
            "nice",
            "system",
            "iowait",
            "steal",
            "idle",
            "total_cpu_utilization",
        ]
    ]


def load_sar_mem_data(log_filepath):
    try:
        df = pd.read_csv(
            log_filepath,
            sep="\\s+",
            skiprows=3,
            skipfooter=1,
            engine="python",
            header=None,
        )
        if df.empty:
            return pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.iloc[0, 1] in ["AM", "PM"]:
        df.columns = [
            "time_part",
            "am_pm",
            "kbmemfree",
            "kbavail",
            "kbmemused",
            "%memused",
            "kbbuffers",
            "kbcached",
            "kbcommit",
            "%commit",
            "kbactive",
            "kbinact",
            "kbdirty",
        ]
        df["time"] = df["time_part"] + " " + df["am_pm"]
    else:
        df.columns = [
            "time",
            "kbmemfree",
            "kbavail",
            "kbmemused",
            "%memused",
            "kbbuffers",
            "kbcached",
            "kbcommit",
            "%commit",
            "kbactive",
            "kbinact",
            "kbdirty",
        ]

    date = get_date(log_filepath)

    def add_date(ts):
        if "-" not in ts:
            return f"{date} {ts}"
        return ts

    df["time"] = df["time"].astype(str).apply(add_date)
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    final_cols = [
        "time",
        "kbmemfree",
        "kbavail",
        "kbmemused",
        "%memused",
        "kbbuffers",
        "kbcached",
        "kbcommit",
        "%commit",
        "kbactive",
        "kbinact",
        "kbdirty",
    ]
    return df[final_cols]


def load_sar_nw_data(log_filepath):
    with open(log_filepath) as f:
        lines = f.readlines()
    data_rows = []
    columns = []
    is_12h_format = False
    lines = lines[3:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Header line
        if re.search(r"\bIFACE\b", line) and not line.startswith("Average"):
            columns = re.split(r"\s+", line)
            if len(columns) > 1 and columns[1] in ("PM", "AM"):
                is_12h_format = True
                columns[0] = "time"  # Rename for clarity
                columns[1] = "am_pm"
            else:
                is_12h_format = False
                columns[0] = "time"
            continue
        # Data line (should match header length)
        if columns and re.match(r"\d{2}:\d{2}:\d{2}", line):
            parts = re.split(r"\s+", line)
            if len(parts) == len(columns):
                data_rows.append(parts)

    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=columns)
    if is_12h_format:
        # For 12h format, combine date, time, and AM/PM.
        df["time"] = df["time"] + " " + df["am_pm"]
        df.drop(columns=["am_pm"], inplace=True)

    date = get_date(log_filepath)

    def add_date(ts):
        if "-" not in ts:
            return f"{date} {ts}"
        return ts

    df["time"] = df["time"].astype(str).apply(add_date)
    df["time"] = pd.to_datetime(df["time"], format="mixed")

    # Convert numeric columns
    for col in df.columns:
        if col not in ["IFACE", "time"]:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df


def plot_sar_cpu(
    df: pd.DataFrame,
    metric: Literal[
        "user", "nice", "system", "iowait", "steal", "idle", "total_cpu_utilization"
    ],
    filename: Path,
):
    """Plot CPU metrics over time, one line per CPU core."""
    plt.figure()
    for label, group in df.groupby("cpu"):
        group = group.sort_values("time")
        processed_group = group.groupby("time")[metric].mean().reset_index()

        # Skip smoothing if not enough data points
        if len(processed_group) <= DEGREE:
            plt.plot(
                processed_group["time"],
                processed_group[metric],
                label=f"cpu: {label}",
                alpha=0.6,
            )
            continue

        # Convert to elapsed seconds and apply spline smoothing
        x = (
            (processed_group["time"] - processed_group["time"].min())
            .dt.total_seconds()
            .to_numpy()
        )
        y = processed_group[metric].to_numpy()

        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=DEGREE)
        y_smooth = spline(x_smooth)

        plt.plot(x_smooth, y_smooth, label=f"cpu: {label}", alpha=0.6)

    plt.xlabel("Timestamp")
    plt.ylabel(metric)
    plt.title(f"Traces of {metric} for each cpu")

    plt.xticks(rotation=45)
    ax = plt.gca()
    ax.set_xlabel("Elapsed Time (s)")

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sar_mem(
    df: pd.DataFrame,
    metric: Literal[
        "kbmemfree",
        "kbavail",
        "kbmemused",
        "%memused",
        "kbbuffers",
        "kbcached",
        "kbcommit",
        "%commit",
        "kbactive",
        "kbinact",
        "kbdirty",
    ],
    filename: Path,
):
    df = df.sort_values("time")

    plt.figure()

    if len(df) <= DEGREE:
        plt.plot(df["time"], df[metric])
    else:
        x = (df["time"] - df["time"].min()).dt.total_seconds().to_numpy()

        # check for duplicates
        unique, counts = np.unique(x, return_counts=True)
        duplicates = unique[counts > 1]
        for val in duplicates:
            dup_indices = np.where(x == val)[0]
            print(f"Duplicates found at indices {dup_indices}")

        y = df[metric].to_numpy()

        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=DEGREE)
        y_smooth = spline(x_smooth)

        plt.plot(x_smooth, y_smooth)

    plt.xlabel("Timestamp")
    plt.ylabel(metric)
    plt.title(f"Traces of {metric}")

    plt.xticks(rotation=45)
    ax = plt.gca()
    ax.set_xlabel("Elapsed Time (s)")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sar_nw(
    df: pd.DataFrame,
    metric: Literal[
        "rxpck/s",
        "txpck/s",
        "rxkB/s",
        "txkB/s",
        "rxcmp/s",
        "txcmp/s",
        "rxmcst/s",
        "%ifutil",
    ],
    filename: Path,
):
    plt.figure()
    for label, group in df.groupby("IFACE"):
        group = group.sort_values("time")
        group = group.groupby("time")[metric].mean().reset_index()

        if len(group) <= DEGREE:
            plt.plot(group["time"], group[metric], label=f"IFACE: {label}", alpha=0.6)
            continue

        x = (group["time"] - group["time"].min()).dt.total_seconds().to_numpy()
        y = group[metric].to_numpy()

        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=DEGREE)
        y_smooth = spline(x_smooth)

        plt.plot(x_smooth, y_smooth, label=f"IFACE: {label}", alpha=0.6)

    plt.xlabel("Timestamp")
    plt.ylabel(metric)
    plt.title(f"Traces of {metric} for each IFACE")

    plt.xticks(rotation=45)
    ax = plt.gca()
    ax.set_xlabel("Elapsed Time (s)")

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
