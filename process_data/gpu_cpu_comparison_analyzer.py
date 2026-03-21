"""
System metrics analyzer.

Compares GPU and CPU utilization from DCGMI and SAR CSV files
to identify resource usage patterns.
"""

import argparse
import glob
import os

import pandas as pd


def analyze_gpu_cpu_comparison(directory_path, threshold):
    try:
        dcgmi_path = glob.glob(os.path.join(directory_path, "dcgmi-local-*.csv"))[0]
        sar_path = glob.glob(os.path.join(directory_path, "sar-cpu-*.csv"))[0]
    except IndexError:
        print(f"Error: Missing files in {directory_path}")
        return

    df_gpu = pd.read_csv(dcgmi_path)
    df_gpu.columns = df_gpu.columns.str.strip()
    df_gpu["timestamp"] = pd.to_datetime(df_gpu["timestamp"]).dt.floor("s")
    gpu_secondly = (
        df_gpu.groupby(["timestamp", "#Entity"])["GPUTL"].mean().reset_index()
    )
    gpu_final = gpu_secondly.groupby("timestamp")["GPUTL"].sum().reset_index()

    df_cpu = pd.read_csv(sar_path)
    df_cpu.columns = df_cpu.columns.str.strip()
    df_cpu["time"] = pd.to_datetime(df_cpu["time"]).dt.floor("s")
    cpu_instants = (
        df_cpu.groupby(["time", "cpu"])["total_cpu_utilization"].mean().reset_index()
    )
    cpu_final = (
        cpu_instants.groupby("time")["total_cpu_utilization"].sum().reset_index()
    )
    cpu_final.rename(columns={"time": "timestamp"}, inplace=True)

    merged = pd.merge(gpu_final, cpu_final, on="timestamp", how="inner")

    res1 = (merged["GPUTL"] > threshold).sum()
    res2 = (merged["total_cpu_utilization"] > threshold).sum()
    res3 = (merged["GPUTL"] > merged["total_cpu_utilization"]).sum()
    res4 = (merged["GPUTL"] < merged["total_cpu_utilization"]).sum()
    res5 = (merged["GPUTL"] == merged["total_cpu_utilization"]).sum()

    print(f"\nResults for Directory: {directory_path}")
    print(f"Threshold used: {threshold}")
    print(f"Timesteps counted: {merged.shape[0]}")
    print("-" * 40)
    print(f"1. Timesteps where GPUTL > {threshold}: {res1}")
    print(f"2. Timesteps where CPU utilization > {threshold}: {res2}")
    print(f"3. Timesteps where GPUTL > CPU utilization: {res3}")
    print(f"4. Timesteps where GPUTL < CPU utilization: {res4}")
    print(f"5. Timesteps where GPUTL = CPU utilization: {res5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze GPU and CPU utilizations from CSV files."
    )
    parser.add_argument(
        "directory", help="Path to the directory containing the CSV files"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=90.0,
        help="Threshold value for comparison (default: 90.0)",
    )

    args = parser.parse_args()
    analyze_gpu_cpu_comparison(args.directory, args.threshold)
