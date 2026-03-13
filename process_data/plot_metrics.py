import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .dcgmi_monitor import ENTITY, load_data_dcgmi, plot_dcgmi
from .header import COLS_, ENERGY, POWER_P50, POWER_P90, POWER_P99
from .sar_monitor import (
    load_sar_cpu_data,
    load_sar_mem_data,
    load_sar_nw_data,
    plot_sar_cpu,
    plot_sar_mem,
    plot_sar_nw,
)
from .vllm_metrics_monitor import load_data_vllm_metrics, plot_vllm_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TIMESTAMP = "timestamp"
TIMESTAMP_OFFSET = "timestamp_offset"
TIME_DIFF = "time_diff"
POWER = "POWER"
ENERGY_kWh = "ENERGY_kWh"
TOTAL_ENERGY_kWh = "TOTAL_ENERGY_kWh"


def parse_plot_metrics_arguments(specify_directory: bool = True) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Process and plot monitoring metrics.")
    parser.add_argument(
        "--process",
        action="store_true",
        help="Run the data processing step (logs to CSVs).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Run the plotting step (CSVs to plots).",
    )
    parser.add_argument(
        "-f",
        "--override",
        action="store_true",
        help="Override existing metric plots (only effective with --plot).",
    )
    if specify_directory:
        args = parser.parse_args(sys.argv[4:])
    else:
        args = parser.parse_args(sys.argv[1:])

    # If neither --process nor --plot is specified, default to running both.
    if not args.process and not args.plot:
        args.process = True
        args.plot = True
    return args


def find_dcgmi_files(log_dir: Path) -> list:
    """Find all dcgmi log files."""
    dcgmi_pattern = re.compile(r"^dcgmi.*\.log$")
    matches = [
        str(p.relative_to(log_dir))
        for p in log_dir.rglob("*")
        if dcgmi_pattern.match(p.name)
    ]
    return sorted(matches)


def find_sar_cpu_files(log_dir: Path) -> list:
    """Find all sar cpu log files."""
    sar_cpu_pattern = re.compile(r"^sar-cpu.*\.log$")
    matches = [
        str(p.relative_to(log_dir))
        for p in log_dir.rglob("*")
        if sar_cpu_pattern.match(p.name)
    ]
    return sorted(matches)


def find_sar_mem_files(log_dir: Path) -> list:
    """Find all sar mem log files."""
    sar_mem_pattern = re.compile(r"^sar-mem.*\.log$")
    matches = [
        str(p.relative_to(log_dir))
        for p in log_dir.rglob("*")
        if sar_mem_pattern.match(p.name)
    ]
    return sorted(matches)


def find_sar_nw_files(log_dir: Path) -> list:
    """Find all sar nw log files."""
    sar_nw_pattern = re.compile(r"^sar-nw.*\.log$")
    matches = [
        str(p.relative_to(log_dir))
        for p in log_dir.rglob("*")
        if sar_nw_pattern.match(p.name)
    ]
    return sorted(matches)


def find_vllm_metrics_files(log_dir: Path) -> list:
    """Find all vllm metrics log files."""
    vllm_metrics_pattern = re.compile(r"^vllm-metrics.*\.log$")
    matches = [
        str(p.relative_to(log_dir))
        for p in log_dir.rglob("*")
        if vllm_metrics_pattern.match(p.name)
    ]
    return sorted(matches)


def process_dcgmi_to_csv(dcgmi_dir: Path, processed_dir: Path, entities: list = None):
    """Loads raw dcgmi logs, calculates metrics, and saves to CSV and JSON."""
    dcgmi_files = find_dcgmi_files(dcgmi_dir)
    if not dcgmi_files:
        logging.warning("No DCGMI log files found.")
        return

    for file_path_str in dcgmi_files:
        file_path = Path(file_path_str)
        logging.info(f"Processing DCGMI log: {file_path.name}")

        base_name = file_path.stem
        target_dir = processed_dir / file_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        csv_path = target_dir / f"{base_name}.csv"
        json_path = target_dir / f"{base_name}_summary.json"

        df = load_data_dcgmi(dcgmi_dir / file_path)
        if entities:
            df = df[df[ENTITY].isin(entities)]

        if df.empty:
            logging.warning(f"DataFrame is empty for {file_path.name}. Skipping.")
            continue

        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])

        active_indices = df.index[df["GPUTL"] > 0.0]
        results = {}
        if not active_indices.empty:
            active_df = df.loc[active_indices[0] : active_indices[-1]].copy()
            results[POWER_P50] = round(np.percentile(active_df[POWER], 50), 3)
            results[POWER_P90] = round(np.percentile(active_df[POWER], 90), 3)
            results[POWER_P99] = round(np.percentile(active_df[POWER], 99), 3)

            min_timestamp = df[TIMESTAMP].min()
            col_ops = {}

            for k, v in COLS_.items():
                if v["op"]:
                    col_ops[k] = v["op"]

            col_ops[TIMESTAMP] = "min"

            df[TIMESTAMP_OFFSET] = (df[TIMESTAMP] - min_timestamp).dt.total_seconds()
            df_grouped_timestamp = (
                df.groupby(TIMESTAMP_OFFSET).agg(col_ops).reset_index()
            )
            df_grouped_timestamp[TIME_DIFF] = df_grouped_timestamp[
                TIMESTAMP_OFFSET
            ].diff()
            df_grouped_timestamp[ENERGY_kWh] = (
                df_grouped_timestamp[POWER] * df_grouped_timestamp[TIME_DIFF]
            ).fillna(0) / (1000.0 * 3600)
            df_grouped_timestamp[TOTAL_ENERGY_kWh] = df_grouped_timestamp[
                ENERGY_kWh
            ].cumsum()
            total_energy_kWh = df_grouped_timestamp[ENERGY_kWh].sum()
            results[ENERGY] = round(total_energy_kWh, 3)
            logging.info(
                f"Total GPU energy for {file_path.name}: {total_energy_kWh} kWh"
            )
        else:
            logging.warning(f"No GPU activity detected in {file_path.name}.")

        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

        df.to_csv(csv_path, index=False)
        logging.info(f"Saved processed DCGMI data to {csv_path}")


def process_sar_to_csv(
    log_dir: Path, processed_dir: Path, find_files_func, load_func, log_type: str
):
    """Generic function to process SAR logs and save them to CSV files."""
    log_files = find_files_func(log_dir)
    if not log_files:
        logging.warning(f"No SAR {log_type} log files found.")
        return

    for file_path_str in log_files:
        file_path = Path(file_path_str)
        logging.info(f"Processing SAR {log_type} log: {file_path.name}")

        target_dir = processed_dir / file_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        csv_path = target_dir / f"{file_path.stem}.csv"
        df = load_func(log_dir / file_path)

        if df.empty:
            logging.warning(f"No data loaded from {file_path.name}. Skipping.")
            continue

        df["time"] = pd.to_datetime(df["time"], format="mixed")
        time_diffs = df["time"].diff()

        # Identify indices where the time difference is negative (i.e., time went backward)
        rollover_indices = time_diffs[time_diffs < pd.Timedelta(0)].index

        # For each rollover point, add one day to all subsequent rows
        for idx in rollover_indices:
            df.loc[idx:, "time"] += pd.Timedelta(days=1)

        df[TIMESTAMP_OFFSET] = (df["time"] - df["time"].min()).dt.total_seconds()

        df.to_csv(csv_path, index=False)
        logging.info(f"Saved processed SAR {log_type} data to {csv_path}")


def process_vllm_metrics_to_csv(vllm_metrics_dir: Path, processed_dir: Path):
    """Loads raw vllm metrics logs, calculates metrics, and saves to CSV and JSON."""
    vllm_metrics_files = find_vllm_metrics_files(vllm_metrics_dir)
    if not vllm_metrics_files:
        logging.warning("No VLLM METRICS log files found.")
        return

    for file_path_str in vllm_metrics_files:
        file_path = Path(file_path_str)
        logging.info(f"Processing VLLM METRICS log: {file_path.name}")

        base_name = file_path.stem
        target_dir = processed_dir / file_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        csv_path = target_dir / f"{base_name}.csv"

        df = load_data_vllm_metrics(vllm_metrics_dir / file_path)

        if df.empty:
            logging.warning(f"DataFrame is empty for {file_path.name}. Skipping.")
            continue

        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
        min_timestamp = df[TIMESTAMP].min()
        df = df.copy()
        df[TIMESTAMP_OFFSET] = (df[TIMESTAMP] - min_timestamp).dt.total_seconds()
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved processed VLLM METRICS data to {csv_path}")


# --- Plotting Functions (CSV -> Plots) ---


def plot_dcgmi_from_csv(processed_dir: Path, plot_dir: Path, override: bool = False):
    """Reads processed DCGMI CSVs and generates plots."""
    csv_files = sorted(
        p.relative_to(processed_dir) for p in processed_dir.rglob("dcgmi*.csv")
    )

    if not csv_files:
        logging.warning("No processed DCGMI csv files found for plotting.")
        return

    dcgmi_metrics = [
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
    ]

    for csv_file in csv_files:
        logging.info(f"Plotting from DCGMI data: {csv_file.name}")
        df = pd.read_csv(processed_dir / csv_file)

        # # The plotting function expects a timedelta column named 'timestamp'
        df[TIMESTAMP] = pd.to_timedelta(df[TIMESTAMP_OFFSET], unit="s")

        metric_plot_subdir = plot_dir / csv_file.parent / csv_file.stem
        metric_plot_subdir.mkdir(parents=True, exist_ok=True)

        for metric in dcgmi_metrics:
            result_file = metric_plot_subdir / f"{metric}.pdf"
            if override or not result_file.exists():
                plot_dcgmi(df, metric, result_file)


def plot_sar_from_csv(
    processed_dir: Path,
    plot_dir: Path,
    log_prefix: str,
    metrics: list,
    plot_func,
    log_type: str,
    override: bool = False,
):
    """Generic function to plot metrics from processed SAR CSV files."""
    csv_files = sorted(
        p.relative_to(processed_dir) for p in processed_dir.rglob(f"{log_prefix}*.csv")
    )
    if not csv_files:
        logging.warning(f"No processed SAR {log_type} csv files found for plotting.")
        return

    for csv_file in csv_files:
        logging.info(f"Plotting from SAR {log_type} data: {csv_file.name}")
        df = pd.read_csv(processed_dir / csv_file)

        # The plotting function expects a timedelta column named 'time'
        df["time"] = pd.to_timedelta(df[TIMESTAMP_OFFSET], unit="s")

        metric_plot_subdir = plot_dir / csv_file.parent / csv_file.stem
        metric_plot_subdir.mkdir(parents=True, exist_ok=True)

        for metric in metrics:
            file_metric_name = metric.replace(
                "/", "-"
            )  # Sanitize metric name for filename
            result_file = metric_plot_subdir / f"{file_metric_name}.pdf"
            if override or not result_file.exists():
                plot_func(df, metric, result_file)


def plot_vllm_metrics_from_csv(
    processed_dir: Path, plot_dir: Path, override: bool = False
):
    """Reads processed VLLM METRICS CSVs and generates plots."""
    csv_files = sorted(
        p.relative_to(processed_dir) for p in processed_dir.rglob("vllm-metrics*.csv")
    )

    if not csv_files:
        logging.warning("No processed VLLM_METRICS csv files found for plotting.")
        return

    vllm_metrics = [
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
    ]

    for csv_file in csv_files:
        logging.info(f"Plotting from VLLM METRICS data: {csv_file.name}")
        df = pd.read_csv(processed_dir / csv_file)

        # # The plotting function expects a timedelta column named 'timestamp'
        df[TIMESTAMP] = pd.to_timedelta(df[TIMESTAMP_OFFSET], unit="s")

        metric_plot_subdir = plot_dir / csv_file.parent / csv_file.stem
        metric_plot_subdir.mkdir(parents=True, exist_ok=True)

        for metric in vllm_metrics:
            result_file = metric_plot_subdir / f"{metric.replace(':', '-')}.pdf"
            if override or not result_file.exists():
                plot_vllm_metrics(df, metric, result_file)


def process_and_plot_metrics(
    metric_dir: Path,
    processed_dir: Path,
    metric_plot_dir: Path,
    config: dict,
    args: argparse.Namespace,
):
    """Main function to orchestrate data processing and plotting."""
    # Ensure output directories exist
    metric_plot_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.process:
        logging.info("--- Running Data Processing Step ---")
        process_dcgmi_to_csv(
            dcgmi_dir=metric_dir,
            processed_dir=processed_dir,
            entities=config.get("model_config", {}).get("vllm", {}).get("gpus"),
        )
        process_sar_to_csv(
            metric_dir, processed_dir, find_sar_cpu_files, load_sar_cpu_data, "CPU"
        )
        process_sar_to_csv(
            metric_dir, processed_dir, find_sar_mem_files, load_sar_mem_data, "Memory"
        )
        process_sar_to_csv(
            metric_dir, processed_dir, find_sar_nw_files, load_sar_nw_data, "Network"
        )
        # process_vllm_metrics_to_csv(
        #     vllm_metrics_dir=metric_dir,
        #     processed_dir=processed_dir,
        # )

    if args.plot:
        logging.info("--- Running Plotting Step ---")
        plot_dcgmi_from_csv(processed_dir, metric_plot_dir, args.override)

        plot_sar_from_csv(
            processed_dir,
            metric_plot_dir,
            "sar-cpu",
            [
                "user",
                "nice",
                "system",
                "iowait",
                "steal",
                "idle",
                "total_cpu_utilization",
            ],
            plot_sar_cpu,
            "CPU",
            args.override,
        )

        plot_sar_from_csv(
            processed_dir,
            metric_plot_dir,
            "sar-mem",
            [
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
            plot_sar_mem,
            "Memory",
            args.override,
        )

        plot_sar_from_csv(
            processed_dir,
            metric_plot_dir,
            "sar-nw",
            [
                "rxpck/s",
                "txpck/s",
                "rxkB/s",
                "txkB/s",
                "rxcmp/s",
                "txcmp/s",
                "rxmcst/s",
                "%ifutil",
            ],
            plot_sar_nw,
            "Network",
            args.override,
        )

        # plot_vllm_metrics_from_csv(processed_dir, metric_plot_dir, args.override)


# Example Usage
if __name__ == "__main__":
    if len(sys.argv) > 3:
        metric_plot_dir = Path(sys.argv[1])
        metric_dir = Path(sys.argv[2])
        config_file = Path(sys.argv[3])
    else:
        print("Warning: No output directory or config specified. Using default paths.")
        metric_plot_dir = Path("./metric_plots")
        metric_dir = Path("./monitoring_logs")
        config_file = Path("./config.json")

    processed_dir = metric_dir / "processed_data"

    try:
        if config_file.suffix in [".yaml", ".yml"]:
            logging.info(f"Loading YAML config from: {config_file}")
            with open(config_file, "r") as f:
                # Use yaml.safe_load() for security
                config_data = yaml.safe_load(f)
        elif config_file.suffix == ".json":
            logging.info(f"Loading JSON config from: {config_file}")
            with open(config_file, "r") as f:
                config_data = json.load(f)
        else:
            logging.error(f"Unsupported configuration file type: {config_file.suffix}")
            sys.exit(1)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_file}")
        sys.exit(1)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        # Handle both JSON and YAML parsing errors
        logging.error(f"Error decoding configuration from {config_file}: {e}")
        sys.exit(1)

    script_args = parse_plot_metrics_arguments()
    process_and_plot_metrics(
        metric_dir, processed_dir, metric_plot_dir, config_data, script_args
    )
