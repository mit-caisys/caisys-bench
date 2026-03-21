import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd
from sar_monitor import load_sar_cpu_data


def analyze_cpu_utilization(df, cpus_str_list, group_name_list):
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    time_diffs = df["time"].diff()
    rollover_indices = time_diffs[time_diffs < pd.Timedelta(0)].index
    for idx in rollover_indices:
        df.loc[idx:, "time"] += pd.Timedelta(days=1)

    for cpus_str, group_name in zip(cpus_str_list, group_name_list):
        pattern = re.compile(r"^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")
        if bool(pattern.match(cpus_str)):
            cpus = set()
            for part in cpus_str.split(","):
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    cpus.update(range(start, end + 1))
                else:
                    cpus.add(int(part))

            df_group = df[df["cpu"].apply(lambda x: int(x) in cpus)]
            result = df_group.groupby("time")["total_cpu_utilization"].sum().mean()
            print(f"Average total CPU utilization for {group_name}: {result:.2f}%")
        else:
            print(f"Unrecognize CPUs string format: {cpus_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze Workflow CPU utilization from CPU metric log file."
    )
    parser.add_argument(
        "directory",
        help="Path to the directory containing cpu metric log file (sar-cpu-xxx.log)",
    )
    parser.add_argument(
        "-c",
        "--cpus",
        type=str,
        nargs="+",
        help="""The list of groups of cpus to process in range (-) and comma (,) format. 
                For example, "1-3,4 6,9-11" for cpus in [1, 2, 3, 4] in the first group 
                and [6, 9, 10, 11] in the second group""",
    )
    parser.add_argument(
        "-g",
        "--groups",
        type=str,
        nargs="+",
        help="The list of group names corresponding to the cpu groups",
    )

    args = parser.parse_args()
    assert len(args.cpus) == len(args.groups)

    log_filepath = glob.glob(os.path.join(args.directory, "sar-cpu-*.log"))[0]
    df = load_sar_cpu_data(Path(log_filepath), ",".join(args.cpus))
    analyze_cpu_utilization(df, args.cpus, args.groups)
