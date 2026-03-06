"""
Score iteration plotting utilities.

Parses log files to extract per-iteration scores and generate plots.
"""

import csv
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def parse_time(line_str):
    timestamp_pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    m = re.match(timestamp_pattern, line_str)
    if not m:
        return None

    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")


def get_elapsed(line_str, start_time):
    current_time = parse_time(line_str)
    if current_time is None:
        return None
    return (current_time - start_time).total_seconds()


def parse_scores_from_log(log_filepath, output_csv_filepath):
    """
    Parses an OpenEvolve log file to extract combined scores ONLY from
    'Evaluated program' lines.

    Matches lines strictly in the format:
    "... Evaluated program <UUID> ... combined_score=<VALUE> ..."

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

    start_time = None
    for line in lines:
        start_time = parse_time(line)
        if start_time:
            break

    if not start_time:
        print(f"Could not determine start time for {log_filepath}")
        return

    times = []
    scores = []

    evaluator_pattern = re.compile(
        r"Evaluated program [a-f0-9\-]+.*combined_score=([\d\.]+)"
    )

    for line in lines:
        match = evaluator_pattern.search(line)

        if match:
            current_score = float(match.group(1))
            elapsed = get_elapsed(line, start_time)

            if elapsed is not None:
                times.append(elapsed)
                scores.append(current_score)

    with open(output_csv_filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time", "score"])
        for time, score in zip(times, scores):
            writer.writerow([time, score])


def plot_score_from_csv(csv_filepath: Path, output_filepath: Path):
    """
    Reads the raw CSV but plots the RUNNING MAXIMUM (Best Score)
    against ITERATION count.
    """
    raw_scores = []
    if not csv_filepath.exists():
        return

    with open(csv_filepath, "r") as csvfile:
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

    plot_scores_utility(
        [iterations], [best_so_far], ["Best Score"], str(output_filepath)
    )


def plot_scores_utility(iterations_lists, scores_lists, titles, output_path):
    fig, ax = plt.subplots()

    for iters, scores, title in zip(iterations_lists, scores_lists, titles):
        ax.plot(iters, scores, label=title, linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Combined Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.7)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


def find_log_files(log_dir: Path) -> list:
    log_pattern = re.compile(r".*\.log$")
    matches = []

    for logs_dir in log_dir.glob("*/logs"):
        for p in logs_dir.rglob("*"):
            if p.is_file() and log_pattern.match(p.name):
                matches.append(str(log_dir / p.relative_to(log_dir)))

    return sorted(matches)


def process_and_parse_scores_iterations_to_csv(
    output_log_dir: Path, processed_dir: Path, metric_dir: Path
):
    log_files = find_log_files(output_log_dir)
    for log_file in log_files:
        # Construct output paths securely
        # Assuming log_file comes from find_log_files relative logic or absolute path
        # Adjust logic to ensure valid path composition

        # If log_file is a relative string from find_log_files logic:
        # We need to act carefully. Based on find_log_files, it returns strings.

        # Re-constructing Path object to handle splitting easily

        # NOTE: This split logic depends heavily on your directory structure.
        # Ensure 'log_file' path depth matches these indices (-3, -1, etc).
        try:
            exp_name = log_file.split("/")[-3]
            log_name = log_file.split("/")[-1]
            stem_name = log_name.split(".")[-2] if "." in log_name else log_name

            output_file = processed_dir / exp_name / f"score_iteration_{stem_name}.csv"
            output_pdf_file = metric_dir / exp_name / f"score_iteration_{stem_name}.pdf"

            parse_scores_from_log(log_file, output_file)
            plot_score_from_csv(output_file, output_pdf_file)
        except IndexError:
            print(f"Skipping file due to path structure mismatch: {log_file}")
            continue
