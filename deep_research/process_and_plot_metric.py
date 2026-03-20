"""
Metrics processing and plotting pipeline for deep research experiments.

Processes raw monitoring data (DCGMI, SAR, vLLM) and generates visualizations.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from paths import METRIC_PLOT_DIR, PROCESSED_METRIC_DIR, RAW_METRIC_DIR, STEP_LOG_DIR

sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_data import (
    parse_plot_metrics_arguments,
    plot_energy,
    plot_timelines_and_p90,
    process_and_plot_metrics,
    process_and_plot_vllm_metric,
    running_power_percentile_plotter,
)

width = 5
height = width / 1.618

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


if __name__ == "__main__":
    PROCESSED_METRIC_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    script_args = parse_plot_metrics_arguments(specify_directory=False)
    process_and_plot_metrics(
        RAW_METRIC_DIR,
        PROCESSED_METRIC_DIR,
        METRIC_PLOT_DIR,
        {"model_config": {"vllm": {"gpus": [1]}}},
        script_args,
    )
    process_and_plot_vllm_metric(RAW_METRIC_DIR, PROCESSED_METRIC_DIR, METRIC_PLOT_DIR)
    plot_energy(PROCESSED_METRIC_DIR, METRIC_PLOT_DIR)
    plot_timelines_and_p90(
        STEP_LOG_DIR, METRIC_PLOT_DIR, "*.csv", ["plan", "action"], "timeline"
    )

    for root, dirs, files in os.walk(PROCESSED_METRIC_DIR):
        for subdir in dirs:
            running_power_percentile_plotter(
                PROCESSED_METRIC_DIR / subdir,
                METRIC_PLOT_DIR / subdir,
                [50, 90, 99],
                [0, 1],
            )
