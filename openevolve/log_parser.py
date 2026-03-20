"""
OpenEvolve log parsing utilities.

Parses log files to extract activity timelines, LLM requests, and evaluation events.
"""

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_log_to_csv(log_filepath, output_csv_filepath):
    """
    Parses the OpenEvolve log file to track active evaluations and LLM requests,
    outputting changes to a CSV file precisely when counts change.

    Args:
        log_filepath (str): Path to the input log file.
        output_csv_filepath (str): Path to the output CSV file.
    """

    # Regex to capture timestamp and relevant log messages
    log_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\,\d{3}) - "  # Timestamp
        r"([\w.]+) - "  # Module
        r"(?:INFO|DEBUG|WARNING|ERROR) - "  # Log Level
        r"(.*)"  # Message
    )

    llm_start_pattern = re.compile(r"API parameters:")
    llm_end_ok_pattern = re.compile(r"API response:")
    llm_end_timeout_pattern = re.compile(r"Timeout on attempt \d+/\d+")
    llm_end_fail_pattern = re.compile(
        r"(?:All \d+ attempts failed|LLM generation failed)"
    )
    eval_start_pattern = re.compile(r"Started process pool with (\d+) processes")

    eval_end_pattern = re.compile(r"Evaluated program")

    events = []
    num_workers = 0
    failed_llm_markers = defaultdict(bool)

    print(f"Reading log file: {log_filepath}")
    try:
        with open(log_filepath, "r") as f:
            for line_num, line in enumerate(f):
                match = log_pattern.match(line)
                if match:
                    timestamp_str, module, message = match.groups()
                    try:
                        timestamp = datetime.strptime(
                            timestamp_str, "%Y-%m-%d %H:%M:%S,%f"
                        )
                    except ValueError:
                        print(f"Warning: Could not parse timestamp: {timestamp_str}")
                        continue

                    failure_marker = f"{timestamp_str}-{line_num}"

                    if "openevolve.llm.openai" in module:
                        if llm_start_pattern.search(message):
                            events.append({"timestamp": timestamp, "type": "LLM_START"})
                            events.append(
                                {"timestamp": timestamp, "type": "EVAL_START_INC"}
                            )

                        elif llm_end_ok_pattern.search(message):
                            # LLM response just ends the LLM part
                            events.append({"timestamp": timestamp, "type": "LLM_END"})

                        elif (
                            llm_end_fail_pattern.search(message)
                            and not failed_llm_markers[failure_marker]
                        ):
                            events.append({"timestamp": timestamp, "type": "LLM_END"})
                            events.append(
                                {"timestamp": timestamp, "type": "EVAL_END_DEC"}
                            )
                            failed_llm_markers[failure_marker] = True

                        elif (
                            llm_end_timeout_pattern.search(message)
                            and not failed_llm_markers[failure_marker]
                        ):
                            # We'll let the 'All attempts failed' message handle the event end
                            pass

                    elif "openevolve.evaluator" in module:
                        if eval_end_pattern.search(message):
                            events.append(
                                {"timestamp": timestamp, "type": "EVAL_END_DEC"}
                            )

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_filepath}")
        return
    except Exception as e:
        print(f"An error occurred while reading the log file: {e}")
        return

    if not events:
        print("No relevant log events found.")
        return

    # Sort events strictly by timestamp
    events.sort(key=lambda x: x["timestamp"])

    output_data = [["timestamp", "active_evaluations", "active_llm_requests"]]
    active_evals = 0
    active_llms = 0
    last_recorded_evals = -1  # Sentinel value
    last_recorded_llms = -1  # Sentinel value
    first_event_processed = False

    print("Processing events...")
    # Process events chronologically
    for event in events:
        current_timestamp = event["timestamp"]
        event_type = event["type"]
        state_changed = False

        # Record initial state (0, 0) at the time of the very first relevant event
        if not first_event_processed:
            timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            # Only add initial state if it's different from the state *after* this first event
            temp_evals = active_evals
            temp_llms = active_llms
            if event_type == "EVAL_START":
                temp_evals = event["value"]
            elif event_type == "LLM_START":
                temp_llms += 1

            if temp_evals != 0 or temp_llms != 0:
                output_data.append([timestamp_str, 0, 0])
                last_recorded_evals = 0
                last_recorded_llms = 0
            first_event_processed = True

        # Update state based on the current event
        if event_type == "LLM_START":
            active_llms += 1
        elif event_type == "LLM_END":
            active_llms = max(0, active_llms - 1)  # Prevent going below zero

        elif event_type == "EVAL_START_INC":
            active_evals += 1

        elif event_type == "EVAL_END_DEC":
            active_evals = max(0, active_evals - 1)

        # Check if the state *after* processing the event has changed from the last recorded state
        if active_evals != last_recorded_evals or active_llms != last_recorded_llms:
            timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            output_data.append([timestamp_str, active_evals, active_llms])
            last_recorded_evals = active_evals
            last_recorded_llms = active_llms

    print(f"Writing output to CSV: {output_csv_filepath}")
    try:
        with open(output_csv_filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(output_data)
        print("CSV file written successfully.")
    except Exception as e:
        print(f"An error occurred while writing the CSV file: {e}")


def find_log_files(log_dir: Path) -> list:
    log_pattern = re.compile(r".*\.log$")
    matches = []

    for logs_dir in log_dir.glob("*/logs"):
        for p in logs_dir.rglob("*"):
            if p.is_file() and log_pattern.match(p.name):
                matches.append(str(log_dir / p.relative_to(log_dir)))

    return sorted(matches)


def process_and_parse_to_csv(output_log_dir: Path, processed_dir: Path):
    log_files = find_log_files(output_log_dir)
    for log_file in log_files:
        output_file = (
            processed_dir
            / log_file.split("/")[-3]
            / f"request_{log_file.split('/')[-1].split('.')[-2]}.csv"
        )
        parse_log_to_csv(log_file, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process OpenEvolve logs to track active evaluations and LLM requests."
    )
    parser.add_argument("log_file", help="Path to the input log file.")
    parser.add_argument(
        "-o",
        "--output",
        default="activity_log_fixed.csv",
        help="Path to the output CSV file (default: activity_log_fixed.csv)",
    )

    args = parser.parse_args()

    parse_log_to_csv(args.log_file, args.output)
