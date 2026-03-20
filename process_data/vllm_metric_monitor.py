"""
vLLM metrics data loading and plotting utilities.

Handles parsing of vLLM Prometheus-format metrics logs.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

TIMESTAMP = "timestamp"
PROPERTY = "property"
PROPERTY_START_TIME = "start_time"
PROPERTY_END_TIME = "end_time"
PROPERTY_REFRESH_FREQ_S = "refresh_freq_s"
DATETIME_FORMAT = "%Y%m%d-%H%M%S"


def load_data_vllm_metrics(log_filepath):
    lines = []
    properties = {}
    line_number = -1
    with open(log_filepath) as file:
        for line in file:
            if line.startswith(
                "# HELP python_gc_objects_collected_total Objects collected during gc"
            ):
                line_number += 1
            elif line.startswith(PROPERTY):
                _, name, val = line.strip().split("=")
                properties[name] = val
            elif not line.startswith("#"):
                lines.append([line_number, line.strip()])

    # Get the header name
    header_idx = {}
    header = []
    for line in lines:
        metric_name = line[1].split()[0]
        if metric_name not in header_idx:
            header_idx[metric_name] = len(header)
            header.append(metric_name)

    start_time = datetime.strptime(properties[PROPERTY_START_TIME], DATETIME_FORMAT)
    end_time = datetime.strptime(properties[PROPERTY_END_TIME], DATETIME_FORMAT)
    refresh_freq_s = int(properties[PROPERTY_REFRESH_FREQ_S])
    print(f"VLLM METRICS duration: {(end_time - start_time).total_seconds()}s")

    # Add timestamp to each line
    data = []
    row = ["0.0"] * len(header)
    timestamp = start_time
    line_number = 0
    for line in lines:
        if line[1].strip() == "":
            continue
        else:
            if line_number != line[0]:
                line_number = line[0]
                row.insert(0, timestamp)
                timestamp += timedelta(seconds=refresh_freq_s)
                assert timestamp <= end_time

                data.append(row)
                row = ["0.0"] * len(header)

            row[header_idx[line[1].split()[0]]] = line[1].split()[-1]

    row.insert(0, timestamp)
    data.append(row)

    # Load into DF
    header.insert(0, TIMESTAMP)
    df = pd.DataFrame(data, columns=header)
    for col in df.columns:
        if col == TIMESTAMP:
            continue
        df[col] = df[col].astype(float)
    return df


def plot_vllm_metrics(
    df: pd.DataFrame,
    metric_prefix: Literal[
        "vllm:gpu_prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_queries_created",
        "vllm:gpu_prefix_cache_hits_total",
        "vllm:gpu_prefix_cache_hits_created",
        "vllm:prompt_tokens_total",
        "vllm:prompt_tokens_created",
        "vllm:generation_tokens_total",
        "vllm:generation_tokens_created",
        "vllm:request_prompt_tokens_bucket",
        "vllm:request_prompt_tokens_count",
        "vllm:request_prompt_tokens_sum",
        "vllm:request_prompt_tokens_created",
        "vllm:request_generation_tokens_bucket",
        "vllm:request_generation_tokens_count",
        "vllm:request_generation_tokens_sum",
        "vllm:request_generation_tokens_created",
        "vllm:iteration_tokens_total_bucket",
        "vllm:iteration_tokens_total_count",
        "vllm:iteration_tokens_total_sum",
        "vllm:iteration_tokens_total_created",
        "vllm:request_max_num_generation_tokens_bucket",
        "vllm:request_max_num_generation_tokens_count",
        "vllm:request_max_num_generation_tokens_sum",
        "vllm:request_max_num_generation_tokens_created",
        "vllm:request_params_max_tokens_bucket",
        "vllm:request_params_max_tokens_count",
        "vllm:request_params_max_tokens_sum",
        "vllm:request_params_max_tokens_created",
        "vllm:time_to_first_token_seconds_bucket",
        "vllm:time_to_first_token_seconds_count",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_created",
        "vllm:time_per_output_token_seconds_bucket",
        "vllm:time_per_output_token_seconds_count",
        "vllm:time_per_output_token_seconds_sum",
        "vllm:time_per_output_token_seconds_created",
    ],
    filename: Path,
):
    DEGREE = 3
    plt.figure()
    for metric in df.columns:
        if not metric.startswith(metric_prefix):
            continue

        if len(df) <= DEGREE:
            plt.plot(df["timestamp"], df[metric], label=metric)
            continue

        x = (df[TIMESTAMP] - df[TIMESTAMP].min()).dt.total_seconds().to_numpy()
        y = df[metric].to_numpy()

        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=DEGREE)
        y_smooth = spline(x_smooth)

        plt.plot(x_smooth, y_smooth, label=metric, alpha=0.6)

    plt.xlabel("Elapsed Time (s)")
    plt.title(f"Traces of {metric_prefix}")
    plt.xticks(rotation=45)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
