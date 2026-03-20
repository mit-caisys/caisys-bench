"""
Energy calculation utilities.

Calculates cumulative energy consumption at specific time offsets from DCGMI data.
"""

import argparse
from pathlib import Path

import pandas as pd


def get_energy_at_offset(csv_path: Path, offset_seconds: float) -> float:
    """
    Calculates the cumulative energy (kWh) consumed from the start
    until a specific time offset.
    """

    df = pd.read_csv(csv_path)
    df_grouped = df.groupby("timestamp_offset")["POWER"].sum().reset_index()

    df_grouped["time_diff"] = df_grouped["timestamp_offset"].diff().fillna(0)
    df_grouped["energy_kWh"] = (df_grouped["POWER"] * df_grouped["time_diff"]) / (
        1000.0 * 3600
    )
    df_grouped["total_energy_kWh"] = df_grouped["energy_kWh"].cumsum()
    relevant_data = df_grouped[df_grouped["timestamp_offset"] <= offset_seconds]

    if relevant_data.empty:
        return 0.0

    energy_val = relevant_data["total_energy_kWh"].iloc[-1]
    return round(float(energy_val), 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate energy up to the offset")
    parser.add_argument("dcgmi_csv_file", help="Path to the dcgmi csv file")
    parser.add_argument(
        "--offset", type=float, default=float("inf"), help="Timestamp offset"
    )
    args = parser.parse_args()
    energy = get_energy_at_offset(args.dcgmi_csv_file, args.offset)
    print(f"Energy is {energy}")
