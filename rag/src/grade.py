import argparse
from pathlib import Path

import pandas as pd

from common.header import (
    ACTUAL_ANSWER_HEADER,
    CORRECT_HEADER,
    EXPECTED_ANSWER_HEADER,
    QUESTION_HEADER,
)
from common.path import RESULT_DIR
from self_rag import create_correctness_grader


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grading script for workflow result")

    parser.add_argument(
        "-f",
        "--override",
        action="store_true",
        help="Override the grade on a result",
    )

    parser.add_argument(
        "-t",
        "--target",
        type=Path,
        default=None,
        help=f"Path to the result csv in {RESULT_DIR}",
    )

    args = parser.parse_args()
    return args


def grade_result(
    result_path: Path, override: bool, check_answer_column: str = CORRECT_HEADER
):
    print(f"Start Grading {result_path}")
    df = pd.read_csv(result_path, index_col=0)

    if check_answer_column in df.columns and not override:
        return

    df[check_answer_column] = ""
    grader = create_correctness_grader(
        {
            "provider": "huggingface",
            "model": "google/gemma-3-27b-it",
            "temperature": 0,
        }
    )

    for idx, row in df.iterrows():
        question = row[QUESTION_HEADER]
        llm_answer = row[ACTUAL_ANSWER_HEADER]
        correct_answer = row[EXPECTED_ANSWER_HEADER]
        output = grader.invoke(
            {
                "question": question,
                "llm_answer": llm_answer,
                "correct_answer": correct_answer,
            }
        )

        df.at[idx, check_answer_column] = (
            True if output.binary_score == "yes" else False  # pyright: ignore
        )

    df.to_csv(result_path)


def find_results(result_dir: Path = RESULT_DIR):
    return [
        file_path
        for file_path in result_dir.iterdir()
        if file_path.is_file() and file_path.suffix == ".csv"
    ]


def main(args: argparse.Namespace):
    target_results = (
        [(RESULT_DIR / args.target).resolve()]
        if args.target != None
        else find_results()
    )
    for result in target_results:
        grade_result(result, args.override)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
