"""
Score comparison plotting utilities.

Combines multiple score evolution plots from different experiments into a single figure.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

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


def plot_score(csv_paths: list[Path], labels: list[str]):
    """
    Plots score over iteration for a list of CSVs.
    """

    for csv_path, label in zip(csv_paths, labels):
        try:
            raw_scores = []
            if not csv_path.exists():
                return

            with open(csv_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        raw_scores.append(float(row["score"]))
                    except (ValueError, KeyError):
                        continue

            if not raw_scores:
                return

            best_so_far = []
            current_max = -float("inf")
            for s in raw_scores:
                if s > current_max:
                    current_max = s
                best_so_far.append(current_max)

            iterations = list(range(len(best_so_far)))

            plt.plot(iterations, best_so_far, label=label, alpha=0.8)

            print(f"Processed {csv_path}")

        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True)

    output_filename = "score_over_iteration.pdf"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score over iteration from CSV logs")
    parser.add_argument(
        "--csv_files", nargs="+", type=Path, help="list of csv files to plot"
    )
    parser.add_argument(
        "--labels", nargs="+", type=str, help="list of corresponding labels"
    )
    args = parser.parse_args()

    plot_score(args.csv_files, args.labels)
