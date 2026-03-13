import argparse
import ast
import asyncio
import copy
import datetime
import json
import logging
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import openai
import pandas as pd
from common.config import CONFIG_DIR, CONFIG_HUGGINGFACE_PATH, CONFIG_OPENAI_PATH
from common.header import (
    ACTUAL_ANSWER_HEADER,
    END_TIME_HEADER,
    EXPECTED_ANSWER_HEADER,
    LOAD_GEN_END_TIME_HEADER,
    LOAD_GEN_START_TIME_HEADER,
    NODES_PATH_HEADER,
    QUESTION_HEADER,
    REQUEST_ID_HEADER,
    RESULT_HEADERS,
    START_TIME_HEADER,
    TOTAL_INPUT_TOKENS_HEADER,
    TOTAL_OUTPUT_TOKENS_HEADER,
)
from common.path import FRAMES_PATH, LOG_DIR, RAW_METRIC_DIR, RESULT_DIR
from common.prefix import BASELINE_PREFIX, POISSON_PREFIX, RESULT_PREFIX
from dotenv import load_dotenv
from self_rag import create_input, create_self_rag, run_self_rag

sys.path.append(str(Path(__file__).resolve().parents[2]))

from load_generator import Request, poisson_load_generator  # pyright: ignore
from monitoring import DCGMI, SAR  # pyright: ignore

CONFIG_PATH = CONFIG_HUGGINGFACE_PATH
# CONFIG_PATH = CONFIG_OPENAI_PATH
NUM_FRAMES_QUESTIONS_USED = 100
NUM_FRAMES_QUESTIONS_VECTORSTORE_DEFAULT = 100
BROKEN_MILVUS_COLLECTION_NAMES = ["openai_split_2000_200_frames_benchmark_37"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark script for RAG workflow")

    parser.add_argument(
        "-b",
        "--baseline",
        action="store_true",
        help="Run baseline result",
    )

    parser.add_argument(
        "-l",
        "--load-generator",
        type=float,
        default=None,
        help="Poisson rate to run with load generator",
    )

    parser.add_argument(
        "-t",
        "--timeline",
        action="store_true",
        help="Write timeline of the experiment",
    )

    parser.add_argument(
        "-n",
        "--num-set-links",
        type=int,
        default=NUM_FRAMES_QUESTIONS_VECTORSTORE_DEFAULT,
        help=f"Number of frames Wiki links included to create the vector database (default: {NUM_FRAMES_QUESTIONS_VECTORSTORE_DEFAULT})",
    )

    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=CONFIG_PATH,
        help=f"Path to the config file in {CONFIG_DIR}",
    )

    args = parser.parse_args()
    assert (
        args.num_set_links >= NUM_FRAMES_QUESTIONS_USED
    ), f"Number of frames Wiki links used must include the first {NUM_FRAMES_QUESTIONS_USED} questions"

    args.file = CONFIG_DIR / args.file
    if not args.file.is_file():
        raise FileNotFoundError(f"The config file '{args.file}' does not exist.")

    return args


def load_config(config_path: Path = CONFIG_PATH):
    """
    Load the json configuration file for the workflow
    """
    with open(config_path, "r") as file:
        config = json.load(file)

    add_optional_arguments(config)
    return config


def add_optional_arguments(config):
    """
    Add optional arguments to the config, if not presented
    """
    config["request_id"] = config.get("request_id", "RAG_request")
    config["verbose"] = config.get("verbose", False)

    # Iteration
    config["iteration"] = config.get("iteration", {})
    config["iteration"]["max_rewrite_iterations"] = config["iteration"].get(
        "max_rewrite_iterations", 3
    )
    config["iteration"]["max_generate_iterations"] = config["iteration"].get(
        "max_generate_iterations", 2
    )

    # Retriever
    config["retriever"]["split"] = config["retriever"].get("split", True)
    config["retriever"]["chunk_size"] = config["retriever"].get("chunk_size", 2000)
    config["retriever"]["chunk_overlap"] = config["retriever"].get("chunk_overlap", 200)
    config["retriever"]["search_type"] = config["retriever"].get(
        "search_type", "similarity"
    )
    config["retriever"]["search_kwargs"] = config["retriever"].get("search_kwargs", {})

    # Model
    config["hallucination_grader"] = config.get(
        "hallucination_grader", config["retrieval_grader"]
    )
    config["answer_grader"] = config.get("answer_grader", config["retrieval_grader"])
    config["question_rewriter"] = config.get("question_rewriter", config["generator"])

    # User Input
    config["documents"] = config.get(
        "documents", {"collection_name": "LangChainCollection", "override": False}
    )


def create_workflow(workflow: str):
    match workflow:
        case "self_rag":
            return create_self_rag()
        case _:
            raise KeyError(f"Cannot find workflow {workflow}")


def run_workflow(workflow, config, args):
    match config["workflow"]:
        case "self_rag":
            return run_self_rag(workflow, config, args.timeline)
        case _:
            raise KeyError(f"Cannot find workflow {config['workflow']}")


def override_frames_config(
    run_type: Literal["baseline", "experiment"],
    config,
    df,
    index: int,
    urls: list[str] = [],
    num_indices: int = NUM_FRAMES_QUESTIONS_USED,
):
    # Q&A
    config["question"] = df.iloc[index]["Prompt"]
    config["answer"] = df.iloc[index]["Answer"]

    # Documents
    config["documents"]["urls"] = (
        urls
        if run_type == "experiment"
        else ast.literal_eval(df.iloc[index]["wiki_links"])
    )
    extension = (
        f"_split_{config['retriever']['chunk_size']}_{config['retriever']['chunk_overlap']}"
        if config["retriever"]["split"]
        else ""
    )
    id = index if run_type == "baseline" else f"0_{num_indices - 1}"
    config["documents"][
        "collection_name"
    ] = f"{config['retriever']['embedding']['provider']}{extension}_frames_benchmark_{id}"

    if config["documents"]["collection_name"] in BROKEN_MILVUS_COLLECTION_NAMES:
        config["documents"]["collection_name"] += "_fix"

    config["request_id"] = f"question_{index}"


def convert_to_summary(request: Request):
    summary = {
        REQUEST_ID_HEADER: request.result["request_id"],
        START_TIME_HEADER: request.result["start_time"],
        END_TIME_HEADER: request.result["end_time"],
        LOAD_GEN_START_TIME_HEADER: request.start_time,
        LOAD_GEN_END_TIME_HEADER: request.end_time,
        NODES_PATH_HEADER: ",".join(request.result["nodes_path"]),
        TOTAL_INPUT_TOKENS_HEADER: sum(request.result["input_tokens"]),
        TOTAL_OUTPUT_TOKENS_HEADER: sum(request.result["output_tokens"]),
        QUESTION_HEADER: request.result["question"],
        ACTUAL_ANSWER_HEADER: request.result["generation"],
        EXPECTED_ANSWER_HEADER: request.result["answer"],
    }
    return summary


def parse_timeline_log(log_filename: str):
    request_map = defaultdict(list)

    pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO: (?P<id>\S+) (?P<type>start|end) (?P<nodename>\w+)"
    )

    with open(LOG_DIR / log_filename, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ts_str = match.group("timestamp")
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
                qid = match.group("id")
                entry_type = match.group("type")
                node = match.group("nodename")
                request_map[qid].append((node, entry_type, ts.timestamp()))

    for qid in request_map:
        request_map[qid].sort(key=lambda x: x[2])
    return request_map


def write_timeline_log(request_map: dict, result_path: Path):
    df = pd.read_csv(result_path, index_col=0)
    for req_id, events in request_map.items():
        counter = {}
        for node, action, ts in events:
            key = f"{node}_{action}"
            idx = counter.get(key, 0)
            col_name = f"{node}_{idx}_{action}"

            row_index = df[df[REQUEST_ID_HEADER] == req_id].index
            if not row_index.empty:
                df.at[row_index[0], col_name] = ts
                counter[key] = idx + 1

    df.to_csv(result_path)


def write_result_frames(
    index: int,
    summary: dict,
    result_path: Path,
    columns_names: list[str] = RESULT_HEADERS,
):
    df = (
        pd.read_csv(result_path, index_col=0)
        if result_path.exists()
        else pd.DataFrame(columns=columns_names)  # pyright: ignore
    )

    if index not in df.index:
        df.loc[index, :] = pd.NA

    for key in columns_names:
        if key in summary:
            df.at[index, key] = summary[key]

    df.sort_index(inplace=True)
    df.to_csv(result_path)


def get_run_type_and_num_indices(args: argparse.Namespace):
    """
    Extract the run_type and num_indices from the input argument
    run_type: type of the run - baseline and experiment
    num_indices: number of frames Wiki links included to create the vector database
    """
    run_type = "baseline" if args.baseline else "experiment"
    num_indices = args.num_set_links
    assert (
        run_type == "baseline" or num_indices >= NUM_FRAMES_QUESTIONS_USED
    ), "Number of indices must be at least the number of questions used"
    if run_type == "baseline":
        num_indices = 1

    return run_type, num_indices


def get_urls(run_type: Literal["baseline", "experiment"], num_indices: int, df):
    """
    URLs to use for vector databases
    if run_type is baseline, return no urls
    """
    urls = []
    if run_type == "experiment":
        for index in range(num_indices):
            links = ast.literal_eval(df.iloc[index]["wiki_links"])
            links = [
                link if link.startswith("https") else f"https://{link}"
                for link in links
            ]
            urls.extend(links)
        urls = list(dict.fromkeys(urls))

    return urls


def get_config_extension(config):
    return (
        f"_split_{config['retriever']['chunk_size']}_{config['retriever']['chunk_overlap']}_{config['retriever']['search_kwargs']['k']}"
        if config["retriever"]["split"]
        else ""
    )


def set_logging(log_filename: str):
    log_path = LOG_DIR / log_filename
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s\n",
        filemode="w",
    )


def run_load_generator(
    run_type: Literal["baseline", "experiment"],
    config,
    df,
    urls: list[str],
    num_indices: int,
    workflow,
    rate: float,
):
    """
    run the load generator experiment, i.e., requests come in poisson arrival rate
    """
    NUM_EPOCHES = 5
    inputs = []
    for index in range(NUM_FRAMES_QUESTIONS_USED):
        override_frames_config(run_type, config, df, index, urls, num_indices)
        problem_config = copy.deepcopy(config)
        inputs.append(create_input(problem_config))

    for epoch in range(NUM_EPOCHES):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"{POISSON_PREFIX}_{config['workflow']}_{timestamp}.log"
        set_logging(log_filename)
        extension = get_config_extension(config)

        base_path = f"{POISSON_PREFIX}_{str(rate).replace('.', '-')}_{config['generator']['provider']}_{config['generator']['model'].replace(':', '-').replace('/', '-')}_{epoch}{extension if config['retriever']['split'] else ''}_{num_indices}"

        output_dir = RAW_METRIC_DIR / base_path
        output_dir.mkdir(parents=True, exist_ok=True)
        with DCGMI(output_dir=output_dir):
            with SAR(output_dir=output_dir):
                requests = asyncio.run(poisson_load_generator(workflow, inputs, rate))

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        result_path = RESULT_DIR / f"{base_path}.csv"
        for index, request in enumerate(requests):
            summary = convert_to_summary(request)
            write_result_frames(index, summary, result_path)

        request_map = parse_timeline_log(log_filename)
        write_timeline_log(request_map, result_path)


def run_sequential(
    run_type: Literal["baseline", "experiment"],
    config,
    df,
    urls: list[str],
    num_indices: int,
    workflow,
):
    """
    run the experiment where each request comes in sequential order
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{BASELINE_PREFIX if run_type == 'baseline' else RESULT_PREFIX}_{config['workflow']}_{timestamp}.log"
    set_logging(log_filename)
    extension = get_config_extension(config)

    result_path = (
        RESULT_DIR
        / f"{BASELINE_PREFIX if run_type == 'baseline' else RESULT_PREFIX}_{config['generator']['provider']}_{config['generator']['model'].replace(':', '-').replace('/', '-')}{extension if config['retriever']['split'] else ''}_{num_indices}.csv"
    )

    for index in range(NUM_FRAMES_QUESTIONS_USED):
        override_frames_config(run_type, config, df, index, urls, num_indices)
        try:
            summary = run_workflow(workflow, config, args)
            write_result_frames(index, summary, result_path)
        except openai.BadRequestError as e:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            print(e)
        except openai.RateLimitError as e:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            print(e)
            time.sleep(60)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    request_map = parse_timeline_log(log_filename)
    write_timeline_log(request_map, result_path)


def main(
    args: argparse.Namespace,
    frames_path: Path = FRAMES_PATH,
):
    run_type, num_indices = get_run_type_and_num_indices(args)
    load_dotenv()
    config = load_config(args.file)
    workflow = create_workflow(config["workflow"])
    df = pd.read_csv(frames_path, sep="\t")
    assert (
        len(df) >= num_indices
    ), f"Number of indices must not exceed the total number of frames questions {len(df)}"

    # urls to use for vector database
    urls = get_urls(run_type, num_indices, df)

    if args.load_generator != None:
        run_load_generator(
            run_type, config, df, urls, num_indices, workflow, args.load_generator
        )
    else:
        run_sequential(run_type, config, df, urls, num_indices, workflow)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
