import os
import sys
from pathlib import Path

from openevolve_log import process_and_parse_to_csv
from path import (
    METRIC_PLOT_DIR,
    OPENEVOLVE_OUTPUT_DIR,
    PROCESSED_METRIC_DIR,
    RAW_METRIC_DIR,
)
from score_plot import process_and_parse_scores_to_csv
from timeline_activity import process_and_plot_activity

sys.path.append(str(Path(__file__).resolve().parents[1]))

from process_data import (
    parse_plot_metrics_arguments,
    plot_energy,
    process_and_plot_metrics,
    process_and_plot_vllm_metric,
    running_power_percentile_plotter,
)

if __name__ == "__main__":
    script_args = parse_plot_metrics_arguments(specify_directory=False)
    process_and_plot_metrics(
        RAW_METRIC_DIR, PROCESSED_METRIC_DIR, METRIC_PLOT_DIR, {}, script_args
    )
    process_and_parse_to_csv(OPENEVOLVE_OUTPUT_DIR, PROCESSED_METRIC_DIR)
    process_and_plot_activity(PROCESSED_METRIC_DIR, METRIC_PLOT_DIR)
    process_and_plot_vllm_metric(RAW_METRIC_DIR, PROCESSED_METRIC_DIR, METRIC_PLOT_DIR)
    process_and_parse_scores_to_csv(
        OPENEVOLVE_OUTPUT_DIR, PROCESSED_METRIC_DIR, METRIC_PLOT_DIR
    )
    plot_energy(PROCESSED_METRIC_DIR, METRIC_PLOT_DIR)

    for root, dirs, files in os.walk(PROCESSED_METRIC_DIR):
        for subdir in dirs:
            running_power_percentile_plotter(
                PROCESSED_METRIC_DIR / subdir,
                METRIC_PLOT_DIR / subdir,
                [50, 90, 99],
                [0, 1],
            )
