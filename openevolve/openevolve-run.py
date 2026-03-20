"""
Entry point script for OpenEvolve benchmark system.
"""

#!/usr/bin/env python
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from paths import RAW_METRIC_DIR

from openevolve.cli import main, parse_args
from openevolve.config import load_config

sys.path.append(str(Path(__file__).resolve().parents[1]))
from monitoring import DCGMI, SAR, VLLMMonitor

TENSOR_PARALLEL_SIZE = os.getenv("TENSOR_PARALLEL_SIZE")

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        cores = config_dict.get("cores")

    num_diverse_programs = config.prompt.num_diverse_programs
    include_artifacts = str(config.prompt.include_artifacts).lower()
    model_names = "_".join([model.name.split("/")[-1] for model in config.llm.models])
    custom_prompt = "_custom-prompt" if config.prompt.template_dir else ""

    output_dir = (
        RAW_METRIC_DIR
        / f"{TENSOR_PARALLEL_SIZE}_{num_diverse_programs}_{include_artifacts}{custom_prompt}_{model_names}_{cores}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with DCGMI(output_dir=output_dir):
        with SAR(output_dir=output_dir):
            with VLLMMonitor(output_dir=output_dir):
                sys.exit(main())
