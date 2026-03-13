import re
from pathlib import Path
from typing import Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

DATETIME_FORMAT = "%H:%M:%S"
DEGREE = 3
N = 10


def load_sar_cpu_data(log_filepath: Path):
    """
    Loads CPU utilization data from a sar log file, with robust handling
    for various common date and time formats.

    Args:
        log_filepath (Path): The path to the sar log file.
        dayfirst (bool): Set to True if the date format is Day/Month/Year
                         (e.g., 31/07/2025). Defaults to False (Month/Day/Year).

    Returns:
        pd.DataFrame: A DataFrame with parsed CPU data.
    """
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

    df["time"] = pd.to_datetime(time_series, format="mixed")

    # Get pinned cpu cores
    cpus_str = log_filepath.parent.name.split("_")[-1]

    # check if the format matches
    pattern = re.compile(r"^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")
    if bool(pattern.match(cpus_str)):
        cpus = set()
        for part in cpus_str.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                cpus.update(range(start, end + 1))
            else:
                cpus.add(int(part))

        df = df[df["cpu"].apply(lambda x: x.isnumeric() and int(x) in cpus)].copy()
    else:
        df = df[(df["cpu"] == "all") | df["cpu"].str.isnumeric()].copy()

    df["idle"] = pd.to_numeric(df["idle"])
    df["total_cpu_utilization"] = 100 - df["idle"]

    final_cols = [
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
    return df[final_cols]


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
    # Identify CPUs that ever reach >= 50% total utilization
    # high_util_cpus = (
    #     df.loc[df["total_cpu_utilization"] >= 50, "cpu"].dropna().unique().tolist()
    # )
    # df = df[df["cpu"].isin(high_util_cpus)]

    plt.figure()
    for label, group in df.groupby("cpu"):
        group = group.sort_values("time")
        processed_group = group.groupby("time")[metric].mean().reset_index()

        if len(processed_group) <= DEGREE:
            plt.plot(
                processed_group["time"],
                processed_group[metric],
                label=f"cpu: {label}",
                alpha=0.6,
            )
            continue

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
