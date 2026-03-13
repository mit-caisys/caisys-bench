#!/usr/bin/env python
"""
Entry point script for OpenEvolve
"""
import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from openevolve.cli import main, parse_args
from openevolve.config import load_config
from path import RAW_METRIC_DIR

sys.path.append(str(Path(__file__).resolve().parents[1]))
from monitoring import DCGMI, SAR, VLLM_METRICS, VLLMMonitor  # pyright: ignore

TENSOR_PARALLEL_SIZE = os.getenv("TENSOR_PARALLEL_SIZE")

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        cores = config_dict.get("cores")

    num_diverse_programs = config.prompt.num_diverse_programs
    include_artifacts = config.prompt.include_artifacts
    model_names = "_".join([model.name.split("/")[-1] for model in config.llm.models])

    output_dir = (
        RAW_METRIC_DIR
        / f"{TENSOR_PARALLEL_SIZE}_{num_diverse_programs}_{include_artifacts}_{model_names}_{cores}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with DCGMI(output_dir=output_dir):
        with SAR(output_dir=output_dir):
            # with VLLM_METRICS(output_dir=output_dir):
            with VLLMMonitor(output_dir=output_dir):
                sys.exit(main())
