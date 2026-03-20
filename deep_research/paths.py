"""
Directory path configuration for deep research experiments.
"""

from pathlib import Path

CWD = Path.cwd()

RAW_METRIC_DIR = CWD / "raw_metric"
PROCESSED_METRIC_DIR = CWD / "processed_metric"
METRIC_PLOT_DIR = CWD / "metric_plot"
AGENT_LOG_DIR = CWD / "agent_log"
STEP_LOG_DIR = CWD / "step_log"
