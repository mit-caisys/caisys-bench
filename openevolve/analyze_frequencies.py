"""
Input frequency analysis utilities for OpenEvolve.

Analyzes how often different input programs are sampled during evolution.
"""

import ast
import csv
import hashlib
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt


class ProgramType(Enum):
    CURRENT = "Current"
    TOP = "Top"
    DIVERSE = "Diverse"
    INSPIRE = "Inspire"


@dataclass
class SampledProgram:
    program_type: ProgramType
    score: float
    code: str


@dataclass
class SampledProgramInfo:
    num_current: int
    num_top: int
    num_diverse: int
    num_inspire: int
    score: float


patterns = {
    ProgramType.TOP: r"Program(?: \d+)\s+\(Score:\s+(?P<score>[\d.]+)\).*?```rust(?P<code>.*?)```",
    ProgramType.DIVERSE: r"Program(?: D\d+)\s+\(Score:\s+(?P<score>[\d.]+)\).*?```rust(?P<code>.*?)```",
    ProgramType.INSPIRE: r"Inspiration(?: \d+)\s+\(Score:\s+(?P<score>[\d.]+),\s+Type:\s+\w+\).*?```rust(?P<code>.*?)```",
}


def get_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def add_program(
    program_hashes: dict[str, SampledProgramInfo], program: SampledProgram
) -> None:
    program_hash = get_hash(program.code)
    if program_hash not in program_hashes:
        program_hashes[program_hash] = SampledProgramInfo(
            num_current=0,
            num_top=0,
            num_diverse=0,
            num_inspire=0,
            score=program.score,
        )
    match program.program_type:
        case ProgramType.CURRENT:
            program_hashes[program_hash].num_current += 1
        case ProgramType.TOP:
            program_hashes[program_hash].num_top += 1
        case ProgramType.DIVERSE:
            program_hashes[program_hash].num_diverse += 1
        case ProgramType.INSPIRE:
            program_hashes[program_hash].num_inspire += 1


def extract_programs(text: str, program_type: ProgramType) -> list[SampledProgram]:
    matches = re.finditer(patterns[program_type], text, flags=re.DOTALL)
    results = []
    for match in matches:
        results.append(
            SampledProgram(
                program_type=program_type,
                score=float(match.group("score")),
                code=match.group("code").strip(),
            )
        )
    return results


def extract_current_programs(text) -> SampledProgram:
    code_pattern = r"Current Program\n```rust(?P<code>.*?)```"
    code_match = re.search(code_pattern, text, flags=re.DOTALL)
    score_pattern = r"Current Program Information\n- Fitness: (?P<score>[\d.]+)"
    score_match = re.search(score_pattern, text, flags=re.DOTALL)
    assert code_match is not None and score_match is not None
    return SampledProgram(
        program_type=ProgramType.CURRENT,
        score=float(score_match.group("score")),
        code=code_match.group("code").strip(),
    )


def parse_log_to_csv(log_filepath, output_csv_filepath):
    """
    Parses the OpenEvolve log file to track input program frequencies.

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

    print(f"Reading log file: {log_filepath}")
    program_hashes: dict[str, SampledProgramInfo] = {}
    try:
        with open(log_filepath, "r") as f:
            for line in f:
                match = log_pattern.match(line)
                if match:
                    _, module, message = match.groups()

                    if "openevolve.llm.openai" in module and message.startswith(
                        "API parameters: "
                    ):
                        # Remove the 'API parameters: ' part
                        dict_request = message.split(":", 1)[1].strip()
                        request = ast.literal_eval(dict_request)

                        # index 0 is system prompt
                        content = request["messages"][1]["content"]
                        for program_type in patterns.keys():
                            sampled_programs = extract_programs(content, program_type)
                            for program in sampled_programs:
                                add_program(program_hashes, program)

                        current_program = extract_current_programs(content)
                        add_program(program_hashes, current_program)

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_filepath}")
        return
    except Exception as e:
        print(f"An error occurred while reading the log file: {e}")
        return

    output_data = [
        [
            "program_hash",
            "num_current",
            "num_top",
            "num_diverse",
            "num_inspire",
            "score",
        ]
    ]
    sorted_program_hashes = sorted(
        program_hashes.items(), key=lambda item: item[1].score, reverse=True
    )
    for program_hash, sampled_program_info in sorted_program_hashes:
        output_data.append(
            [
                program_hash,
                sampled_program_info.num_current,
                sampled_program_info.num_top,
                sampled_program_info.num_diverse,
                sampled_program_info.num_inspire,
                sampled_program_info.score,
            ]
        )

    print(f"Writing output to CSV: {output_csv_filepath}")
    try:
        with open(output_csv_filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(output_data)
        print("CSV file written successfully.")
    except Exception as e:
        print(f"An error occurred while writing the CSV file: {e}")


def plot_input_frequencies(output_csv_filepath, pdf_filepath):
    print(f"Reading CSV file: {output_csv_filepath}")
    counters: dict[int, list[float]] = defaultdict(list[float])
    try:
        with open(output_csv_filepath, "r") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            if header != [
                "program_hash",
                "num_current",
                "num_top",
                "num_diverse",
                "num_inspire",
                "score",
            ]:
                print(f"Warning: Unexpected CSV header: {header}")
                print("Expected: ['program_hash', 'frequency']")

            for row in reader:
                number_of_times_sampled = (
                    int(row[1]) + int(row[2]) + int(row[3]) + int(row[4])
                )
                counters[number_of_times_sampled].append(float(row[5]))

    except FileNotFoundError:
        print(f"Error: CSV file not found at {output_csv_filepath}")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    sorted_counters = sorted(counters.keys())
    counts = [len(counters[k]) for k in sorted_counters]

    _, ax1 = plt.subplots()

    ax1.bar(
        sorted_counters,
        counts,
        color="skyblue",
        edgecolor="navy",
        alpha=0.5,
        label="Number of Programs",
    )
    ax1.set_xlabel("Number of Times Sampled")
    ax1.set_ylabel("Number of Programs")
    ax1.tick_params(axis="y")

    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    plt.savefig(pdf_filepath)
    plt.close()
    print(f"Saving pdf plot: {pdf_filepath}")


def find_log_files(log_dir: Path) -> list:
    log_pattern = re.compile(r".*\.log$")
    matches = []

    for logs_dir in log_dir.glob("*/logs"):
        for p in logs_dir.rglob("*"):
            if p.is_file() and log_pattern.match(p.name):
                matches.append(str(log_dir / p.relative_to(log_dir)))

    return sorted(matches)


def process_and_plot_input_frequencies(
    output_log_dir: Path, processed_dir: Path, metric_dir: Path
):
    log_files = find_log_files(output_log_dir)
    for log_file in log_files:
        output_file = (
            processed_dir
            / log_file.split("/")[-3]
            / f"input_frequencies_{log_file.split('/')[-1].split('.')[-2]}.csv"
        )
        plot_file = (
            metric_dir
            / log_file.split("/")[-3]
            / f"input_frequencies_{log_file.split('/')[-1].split('.')[-2]}.pdf"
        )
        parse_log_to_csv(log_file, output_file)
        plot_input_frequencies(output_file, plot_file)
