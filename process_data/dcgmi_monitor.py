import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

TIMESTAMP = "timestamp"
TIMESTAMP_OFFSET = "timestamp_offset"
TIME_DIFF = "time_diff"
POWER = "POWER"
ENERGY_kWh = "ENERGY_kWh"
TOTAL_ENERGY_kWh = "TOTAL_ENERGY_kWh"
NODE = "node"
ENTITY = "#Entity"
PROPERTY = "property"
PROPERTY_START_TIME = "start_time"
PROPERTY_END_TIME = "end_time"
PROPERTY_REFRESH_FREQ_MS = "refresh_freq_ms"
DATETIME_FORMAT = "%Y%m%d-%H%M%S"


COLS_ = {
    "GPUTL": {"ymax": 110, "label": "GPU Util. (%)", "op": "mean"},
    "DRAMA": {"ymax": None, "label": "DRAM Active", "op": "mean"},
    "SMACT": {"ymax": None, "label": "SM Active", "op": "mean"},
    "SMOCC": {"ymax": None, "label": "SM Occupancy", "op": "mean"},
    "POWER": {"ymax": None, "label": "Total GPU Power (W)", "op": "sum"},
    "NVLTX": {"ymax": None, "label": "NVLINK (NVLTX)", "op": "sum"},
    "NVLRX": {"ymax": None, "label": "NVLINK (NVLRX)", "op": "sum"},
    "TOTAL_ENERGY_kWh": {
        "ymax": None,
        "label": "GPU Energy (kWh)",
        "op": None,
    },
}


def get_full_path_without_extension(file_path):
    return str(Path(file_path).resolve().with_suffix(""))


def load_data_dcgmi(log_filepath, entities_filter=None):
    # Load all lines and filter out the lines that start with "ID"
    with open(log_filepath) as file:
        lines = [
            line.replace("GPU ", "").strip()
            for line in file
            if not (line.startswith("ID"))
        ]

    # Get the header name
    header = lines[0].split()

    # Parse properties
    properties = {}
    for line in lines:
        if line.startswith(PROPERTY):
            _, name, val = line.split("=")
            properties[name] = val
    start_time = datetime.strptime(properties[PROPERTY_START_TIME], DATETIME_FORMAT)
    end_time = datetime.strptime(properties[PROPERTY_END_TIME], DATETIME_FORMAT)
    refresh_freq_ms = int(properties[PROPERTY_REFRESH_FREQ_MS])
    print(f"DCGMI duration: {(end_time - start_time).total_seconds()}s")

    # Add timestamp to each line
    data = []
    timestamps = {}
    for line in lines:
        if line.strip() == "":
            continue
        elif "dmon was stopped" in line:
            break
        elif line.startswith(ENTITY):
            continue
        elif line.startswith(PROPERTY):
            continue
        else:
            try:
                row = line.split()
                entity = row[0]
                if entities_filter:
                    if int(entity) not in entities_filter:
                        continue
                    entity = int(entity)
                row[0] = entity
                if entity in timestamps:
                    timestamps[entity] += timedelta(milliseconds=refresh_freq_ms)
                else:
                    timestamps[entity] = start_time
                assert timestamps[entity] <= end_time
                row.insert(0, properties[NODE])
                row.insert(0, timestamps[entity])
                data.append(row)
            except Exception as e:
                print(line)
                raise (e)

    # Load into DF
    header.insert(0, NODE)
    header.insert(0, TIMESTAMP)
    df = pd.DataFrame(data, columns=header)
    for col in df.columns:
        if col == ENTITY or col == NODE or col == TIMESTAMP:
            continue
        df[col] = df[col].replace("N/A", np.nan)
        df[col] = df[col].astype(float)
    return df


def get_properties(log_filepath):
    properties = {}
    with open(log_filepath) as file:
        for line in file:
            line = line.strip()
            if line.startswith(PROPERTY):
                _, name, val = line.split("=")
                properties[name] = val
    return properties


def get_start_end_times_dcgmi_logs(log_filepath):
    properties = get_properties(log_filepath)
    start_time = datetime.strptime(properties[PROPERTY_START_TIME], DATETIME_FORMAT)
    end_time = datetime.strptime(properties[PROPERTY_END_TIME], DATETIME_FORMAT)
    return start_time, end_time


def write_per_gpu_csv(df: pd.DataFrame, columns: list, filepath_suffix: str):
    """
    Writes a separate CSV file for each (entity, column) pair containing 'timestamp' and the column.

    Parameters:
    - df: pandas DataFrame containing at least 'timestamp' and '#Entity' columns.
    - columns: List of column names to export with 'timestamp'.
    - filepath_suffix: Suffix path of the outputs.
    """

    required_columns = ["timestamp", "#Entity"] + columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    for entity in df["#Entity"].unique():
        entity_df = df[df["#Entity"] == entity]
        for column in columns:
            output_df = entity_df[["timestamp", column]]
            filename = f"{filepath_suffix}_gpu_{entity}_{column}.csv"
            output_df.to_csv(filename, index=False)
            print(f"Wrote: {filename}")


def _process_logs(data: List[Dict[str, str]]) -> float:
    """Process dcgmi logs and create .csv files of appropriate columns."""
    dfs = []
    for datum in data:
        log_filepath = datum["log_filepath"]
        print(f"Processing: {log_filepath}")
        entities_filter = datum.get("entities_filter", None)
        if entities_filter:
            print(f"Filtering entities for {log_filepath}: {entities_filter}")
        df = load_data_dcgmi(log_filepath, entities_filter)
        print(f"{log_filepath} entities:", sorted(df[ENTITY].unique()))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(df)

    min_timestamp = df[TIMESTAMP].min()
    col_ops = {}
    for k, v in COLS_.items():
        if v["op"]:
            col_ops[k] = v["op"]
    col_ops[TIMESTAMP] = "min"

    # Compute energy.
    df[TIMESTAMP_OFFSET] = (df[TIMESTAMP] - min_timestamp).dt.total_seconds() * 1000.0
    df_grouped_timestamp = df.groupby(TIMESTAMP_OFFSET).agg(col_ops).reset_index()
    df_grouped_timestamp[TIME_DIFF] = df_grouped_timestamp[TIMESTAMP_OFFSET].diff()
    df_grouped_timestamp[ENERGY_kWh] = (
        df_grouped_timestamp[POWER] * df_grouped_timestamp[TIME_DIFF]
    ).fillna(0) / (1000.0 * 1000.0 * 3600)
    df_grouped_timestamp[TOTAL_ENERGY_kWh] = df_grouped_timestamp[ENERGY_kWh].cumsum()
    total_energy_kWh = df_grouped_timestamp[ENERGY_kWh].sum()
    print(f"Total GPU energy: {total_energy_kWh} kWh")

    root_file = get_full_path_without_extension(log_filepath)
    for col, properties in COLS_.items():
        to_plot = properties.get("to_plot", True)
        if not to_plot:
            continue
        x = df_grouped_timestamp[TIMESTAMP]
        y = df_grouped_timestamp[col]
        xlabel = "Time"
        ylabel = properties.get("label", None)
        filename = f"{root_file}_{col}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([xlabel, ylabel])
            writer.writerows(zip(x, y))
        print(f"Created: {filename}")

    cols = list(COLS_.keys())
    cols.remove(TOTAL_ENERGY_kWh)
    write_per_gpu_csv(df, cols, root_file)

    return total_energy_kWh


def process_dcgmi_logs(logs_paths: List[str], gpus: Optional[List[int]] = None):
    input_data = []
    for log_path in logs_paths:
        input_data.append({"log_filepath": log_path, "entities_filter": gpus})
    return _process_logs(input_data)


def plot_dcgmi(
    df: pd.DataFrame,
    metric: Literal[
        "POWER",
        "POWINST",
        "GPUTL",
        "MCUTL",
        "VMUSG",
        "SMACT",
        "SMOCC",
        "TENSO",
        "DRAMA",
        "FP64A",
        "FP32A",
        "FP16A",
        "PCITX",
        "PCIRX",
        "NVLTX",
        "NVLRX",
        "CPUUT",
    ],
    filename: Path,
):
    # df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["node_entity"] = df["node"].astype(str) + " " + df["#Entity"].astype(str)

    plt.figure()
    DEGREE = 3
    for label, group in df.groupby("node_entity"):
        group = group.sort_values("timestamp")

        if len(group) <= DEGREE:
            plt.plot(
                group["timestamp"], group[metric], label=f"node: {label}", alpha=0.6
            )
            continue

        x = (
            (group["timestamp"] - group["timestamp"].min())
            .dt.total_seconds()
            .to_numpy()
        )
        y = group[metric].to_numpy()

        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=DEGREE)
        y_smooth = spline(x_smooth)

        t0 = group["timestamp"].min()
        # timestamps_smooth = pd.to_datetime(t0 + pd.to_timedelta(x_smooth, unit="s"))

        plt.plot(x_smooth, y_smooth, label=f"node: {label}", alpha=0.6)

    plt.xlabel("Timestamp")
    plt.ylabel(metric)
    plt.title(f"Traces of {metric} for each node and entity")
    plt.xticks(rotation=45)

    # N = 10
    # if len(df["timestamp"].unique()) > N:
    #     ticks_to_use = pd.to_datetime(
    #         pd.Series(df["timestamp"].unique()).sort_values()
    #     )[:: max(1, len(df["timestamp"].unique()) // N)]
    #     plt.gca().set_xticks(ticks_to_use)

    ax = plt.gca()
    ax.set_xlabel("Elapsed Time (s)")
    # time_format = mdates.DateFormatter("%H:%M:%S")
    # ax.xaxis.set_major_formatter(time_format)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
