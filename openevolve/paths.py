"""
Directory path configuration for OpenEvolve experiments.
"""

from pathlib import Path

CWD = Path.cwd()

RAW_METRIC_DIR = CWD / "raw_metric"
PROCESSED_METRIC_DIR = CWD / "processed_metric"
METRIC_PLOT_DIR = CWD / "metric_plot"
OPENEVOLVE_OUTPUT_DIR = CWD / "src" / "openevolve_output"

if __name__ == "__main__":
    RAW_METRIC_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_METRIC_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_PLOT_DIR.mkdir(parents=True, exist_ok=True)
