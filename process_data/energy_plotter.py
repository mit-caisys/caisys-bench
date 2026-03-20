"""
Energy and power plotting utilities.

Generates bar charts for energy consumption and running percentile plots
for power metrics from DCGMI monitoring data.
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_energy(input_dir, output_dir):
    """
    Plot energy consumption bar chart from JSON summary files.

    Reads DCGMI summary JSON files and creates:
    - Bar chart of total energy per folder
    - Grouped bar chart of power percentiles (P50, P90, P99)
    """
    complete_data = []
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                folder_name = os.path.basename(dirpath)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if all(
                            k in data
                            for k in ["POWER_P50", "POWER_P90", "POWER_P99", "ENERGY"]
                        ):
                            data["folder_name"] = folder_name
                            complete_data.append(data)
                        else:
                            print(
                                f"Skipping {file_path}: missing one or more required keys."
                            )
                except json.JSONDecodeError:
                    print(f"Skipping {file_path}: not a valid JSON file.")
                except Exception as e:
                    print(f"An error occurred with file {file_path}: {e}")

    if not complete_data:
        print("No data found. Exiting.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.DataFrame(complete_data)

    df = df.sort_values("folder_name").reset_index(drop=True)

    plt.figure(figsize=(12, 7))
    energy_df = df
    plt.bar(energy_df["folder_name"], energy_df["ENERGY"], color="skyblue")
    plt.xlabel("Folder Name", fontsize=12)
    plt.ylabel("Energy", fontsize=12)
    plt.title("Energy by Folder", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # Updated save path
    energy_plot_path = os.path.join(output_dir, "energy_plot.pdf")
    plt.savefig(energy_plot_path)
    plt.close()
    print(f"Energy plot saved as '{energy_plot_path}'")

    # --- Plotting Power Percentiles ---
    power_df = df[["folder_name", "POWER_P50", "POWER_P90", "POWER_P99"]]
    power_df = power_df.set_index("folder_name")
    power_df.plot(kind="bar", figsize=(15, 8), width=0.8)
    plt.xlabel("Folder Name", fontsize=12)
    plt.ylabel("Power", fontsize=12)
    plt.title("Power Percentiles (P50, P90, P99) by Folder", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Percentiles")
    plt.tight_layout()
    # Updated save path
    power_plot_path = os.path.join(output_dir, "power_percentiles_plot.pdf")
    plt.savefig(power_plot_path)
    plt.close()
    print(f"Power percentiles plot saved as '{power_plot_path}'")


def running_power_percentile_plotter(input_dir, output_dir, percentiles, entities):

    csv_file = None
    for f in os.listdir(input_dir):
        # This condition now specifically looks for the dcgmi file
        if f.endswith(".csv") and "dcgmi" in f:
            csv_file = os.path.join(input_dir, f)
            break

    if csv_file is None:
        print(f"Error: No 'dcgmi' CSV file found in '{input_dir}'")
        return

    print(f"Processing file: {csv_file}")

    entities_to_plot = entities if isinstance(entities, list) else [entities]

    try:
        df = pd.read_csv(csv_file)
        df.rename(columns={"#Entity": "Entity"}, inplace=True)
        # Filter for only the entities we want to plot
        df = df[df["Entity"].isin(entities_to_plot)].copy()
    except Exception as e:
        print(f"Error reading or processing CSV file: {e}")
        return

    if df.empty:
        print(
            f"Error: None of the specified entities {entities_to_plot} found in the data."
        )
        return
    for percentile in percentiles:
        quantile_value = percentile / 100.0
        df[f"running_percentile_{percentile}"] = df.groupby("Entity")[
            "POWER"
        ].transform(lambda x: x.expanding().quantile(quantile_value))

    # --- 4. Plot the results for each entity ---
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(14, 7))

    ax = plt.gca()

    plotted_entities = []
    for entity_id in entities_to_plot:
        entity_df = df[df["Entity"] == entity_id]
        if not entity_df.empty:
            plotted_entities.append(entity_id)
            # Plot the running percentile line
            for percentile in percentiles:
                ax.plot(
                    entity_df["timestamp_offset"],
                    entity_df[f"running_percentile_{percentile}"],
                    label=f"Entity {entity_id} Running {percentile}th Percentile",
                    linewidth=2,
                )
            # Plot the instantaneous power as a lighter, background line
            ax.plot(
                entity_df["timestamp_offset"],
                entity_df["POWER"],
                label=f"Entity {entity_id} Instantaneous Power",
                alpha=0.3,
            )

    ax.set_title(
        f"Running Percentiles of Power for Entities {plotted_entities}",
        fontsize=16,
    )
    ax.set_xlabel("Time Offset (seconds)", fontsize=12)
    ax.set_ylabel("Power (W)", fontsize=12)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    entity_str = "_".join(map(str, plotted_entities))
    output_filename = f"running_percentiles_entities_{entity_str}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved successfully to '{output_path}'")


if __name__ == "__main__":
    input_root_dir = "/home/cerangel/comp-ai-bench/openevolve/circle_packing/processed_metric/1_0_False"
    output_plots_dir = (
        "/home/cerangel/comp-ai-bench/openevolve/circle_packing/processed_metric/energy"
    )

    running_power_percentile_plotter(input_root_dir, output_plots_dir, 90, 0)
