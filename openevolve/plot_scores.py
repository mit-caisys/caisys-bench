"""
Score plotting utilities for OpenEvolve experiments.

Parses log files to extract evolution scores and generates plots.
"""

import csv
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def parse_time(line_str):
    timestamp_pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    m = re.match(timestamp_pattern, line_str)
    if not m:
        print(f"Could not parse line: {line_str}")
        raise RuntimeError

    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")


def get_elapsed(line_str, start_time):
    return (parse_time(line_str) - start_time).total_seconds()


def parse_scores_from_log(log_filepath, output_csv_filepath):
    """
    Parses an OpenEvolve log file to extract the best-so-far combined score at each iteration.

    This function identifies improvements by looking for the "New best solution found" message
    and then retrieves the corresponding score.

    Args:
        log_filepath (str): The file path to the log file.
        output_csv_filepath: The file path to the parsed csv file.
    """

    out_path = Path(output_csv_filepath)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(log_filepath, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {log_filepath}")
        return

    if not lines:
        return

    start_time = parse_time(lines[0])
    times = []
    scores = []

    current_best = -float("inf")
    last_seen_score = None

    for i, line in enumerate(lines):
        score_match = re.search(r"combined_score=([\d\.]+)", line)
        if score_match:
            last_seen_score = float(score_match.group(1))

        explicit_update_match = re.search(r"combined_score: [\d\.]+ -> ([\d\.]+)", line)

        is_initial = "Set initial best program" in line
        is_new_best = "New best solution found" in line or explicit_update_match

        if is_initial or is_new_best:
            elapsed = get_elapsed(line, start_time)
            candidate_score = None

            if explicit_update_match:
                candidate_score = float(explicit_update_match.group(1))

            elif is_new_best and i > 0:
                prev_line = lines[i - 1]
                prev_match = re.search(r"combined_score=([\d\.]+)", prev_line)
                if prev_match:
                    candidate_score = float(prev_match.group(1))

            if candidate_score is None and last_seen_score is not None:
                candidate_score = last_seen_score

            if candidate_score is not None and elapsed is not None:
                if candidate_score > current_best:
                    if scores:
                        times.append(elapsed)
                        scores.append(current_best)

                    current_best = candidate_score
                    times.append(elapsed)
                    scores.append(current_best)

    last_elapsed = get_elapsed(lines[-1], start_time)
    if last_elapsed is not None and scores:
        times.append(last_elapsed)
        scores.append(scores[-1])

    with open(output_csv_filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time", "score"])
        for time, score in zip(times, scores):
            writer.writerow([time, score])


def plot_scores(
    times_lists: List[List[float]],
    scores_lists: List[List[float]],
    titles: List[str],
    output_path: str,
):
    """
    Plots one or more series of combined scores against ELAPSED TIME.

    Args:
        times_lists (List[List[float]]):
            A list where each element is a list of elapsed seconds.
        scores_lists (List[List[float]]):
            A list where each element is a list of scores.
        titles (List[str]):
            A list of string labels for each series.
        output_path (str):
            The full file path to save the plot.
    """
    if len(scores_lists) != len(titles) or len(times_lists) != len(scores_lists):
        print("Error: Mismatch in number of lists (times, scores, titles).")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    for times, scores, title in zip(times_lists, scores_lists, titles):
        if not times or not scores:
            print(f"Warning: No data found for {title}, skipping.")
            continue
        ax.plot(times, scores, marker=".", linestyle="-", label=title)

    ax.set_title("Evolution of Combined Score Over Time", fontsize=16)
    ax.set_xlabel("Elapsed Time (seconds)", fontsize=12)
    ax.set_ylabel("Best Combined Score", fontsize=12)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path)
        print(f"Plot successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")

    plt.close(fig)


def plot_score_from_csv(csv_filepath: Path, output_filepath: Path):
    """
    Reads the activity log CSV, plots active evaluations and LLM requests over time,
    and saves the plot as a PDF.

    Args:
        csv_filepath: Path to the input CSV file.
        output_filepath: Path to the output pdf.
    """
    times = []
    scores = []
    with open(csv_filepath, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            times.append(float(row["time"]))
            scores.append(float(row["score"]))

    plot_scores([times], [scores], ["best score"], str(output_filepath))


def find_log_files(log_dir: Path) -> list:
    log_pattern = re.compile(r".*\.log$")
    matches = []

    for logs_dir in log_dir.glob("*/logs"):
        for p in logs_dir.rglob("*"):
            if p.is_file() and log_pattern.match(p.name):
                matches.append(str(log_dir / p.relative_to(log_dir)))

    return sorted(matches)


def process_and_parse_scores_to_csv(
    output_log_dir: Path, processed_dir: Path, metric_dir: Path
):
    log_files = find_log_files(output_log_dir)
    for log_file in log_files:
        output_file = (
            processed_dir
            / log_file.split("/")[-3]
            / f"score_{log_file.split('/')[-1].split('.')[-2]}.csv"
        )
        output_pdf_file = (
            metric_dir
            / log_file.split("/")[-3]
            / f"score_{log_file.split('/')[-1].split('.')[-2]}.pdf"
        )
        parse_scores_from_log(log_file, output_file)
        plot_score_from_csv(output_file, output_pdf_file)
