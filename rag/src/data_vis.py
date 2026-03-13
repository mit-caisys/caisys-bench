import sys
from pathlib import Path

from common.path import RESULT_DIR, RESULT_PLOT_DIR

sys.path.append(str(Path(__file__).resolve().parents[2]))

from process_data import analyze_runs_cdf, plot_accuracies, plot_timelines_and_p90

if __name__ == "__main__":
    nodes_to_plot_latency = ["retrieve_0", "generate_0"]
    nodes_to_plot = ["total_input_tokens", "total_output_tokens"]
    timeline_nodes = ["retrieve_0", "generate_0"]

    for node in nodes_to_plot_latency:
        analyze_runs_cdf(
            RESULT_DIR,
            RESULT_PLOT_DIR,
            "*.csv",
            node,
            latency=True,
        )

    for node in nodes_to_plot:
        analyze_runs_cdf(
            RESULT_DIR,
            RESULT_PLOT_DIR,
            "*.csv",
            node,
        )

    plot_accuracies(RESULT_DIR, RESULT_PLOT_DIR, "*.csv")

    plot_timelines_and_p90(RESULT_DIR, RESULT_PLOT_DIR, "*.csv", timeline_nodes)
