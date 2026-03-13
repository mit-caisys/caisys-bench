import sys
from pathlib import Path

from common.path import METRIC_PLOT_DIR, PROCESSED_METRIC_DIR, RAW_METRIC_DIR

sys.path.append(str(Path(__file__).resolve().parents[2]))

from process_data import parse_plot_metrics_arguments, process_and_plot_metrics

if __name__ == "__main__":
    script_args = parse_plot_metrics_arguments(specify_directory=False)
    process_and_plot_metrics(
        RAW_METRIC_DIR, PROCESSED_METRIC_DIR, METRIC_PLOT_DIR, {}, script_args
    )
