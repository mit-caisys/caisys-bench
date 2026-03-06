"""
Deep Research agent runner.

Main entry point for running AI agent experiments with web search,
document inspection, and visual QA capabilities.
"""

import argparse
import csv
import os
import re
import sys
import threading
from pathlib import Path

import yaml
from box import Box
from dotenv import load_dotenv
from huggingface_hub import login
from paths import AGENT_LOG_DIR, RAW_METRIC_DIR, STEP_LOG_DIR
from rich.console import Console
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import create_visualizer_tool
from smolagents import GoogleSearchTool  # InferenceClientModel,
from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent

sys.path.append(str(Path(__file__).resolve().parents[1]))
from monitoring import DCGMI, SAR, VLLMMonitor

load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        default="configs/config.yaml",
        help="path to configuration file",
    )

    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=0,
        help="question index",
    )

    return parser.parse_args()


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent(llm):
    model_params = {
        "model_id": llm.model,
        "api_base": llm.api_base,
        "api_key": "EMPTY",
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": llm.max_completion_tokens,
        "temperature": llm.temperature,
    }
    model = LiteLLMModel(**model_params)

    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    WEB_TOOLS = [
        GoogleSearchTool(provider="serpapi"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=llm.tool_calling_agent.max_steps,
        verbosity_level=llm.tool_calling_agent.verbosity_level,
        planning_interval=llm.tool_calling_agent.planning_interval,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"][
        "task"
    ] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    manager_agent = CodeAgent(
        model=model,
        tools=[create_visualizer_tool(llm), TextInspectorTool(model, text_limit)],
        executor_kwargs={"timeout_seconds": 60},
        max_steps=llm.code_agent.max_steps,
        verbosity_level=llm.code_agent.verbosity_level,
        additional_authorized_imports=["*"],
        planning_interval=llm.code_agent.planning_interval,
        managed_agents=[text_webbrowser_agent],
    )

    return manager_agent


def create_safe_filename(input_string, length=40):
    subset = input_string[:length]
    subset = subset.lower()
    subset = subset.replace(" ", "_")
    safe_name = re.sub(r"[^a-z0-9_]", "", subset)
    safe_name = re.sub(r"_+", "_", safe_name).strip("_")

    return safe_name


class DualConsole:
    def __init__(self, filename):
        self.file_console = Console(
            file=open(filename, "a", encoding="utf-8", errors="replace"),
            force_terminal=False,  # Disable terminal sequences for file
            width=120,  # Wider width for file output
        )
        # Keep terminal console with default settings
        self.std_console = Console()

    def print(self, *args, **kwargs):
        try:
            self.file_console.print(*args, **kwargs, highlight=False)
            self.std_console.print(*args, **kwargs)
        except UnicodeEncodeError as e:
            error_msg = f"Unicode handling error: {str(e)}"
            self.file_console.print(error_msg, style="bold red")
            self.std_console.print(error_msg, style="bold red")


def main():
    args = parse_args()
    with open(args.config, "r") as c:
        config = Box(yaml.safe_load(c))

    question = config.questions[args.index]

    # Setup dual console to save log file
    output_dir_base = f"{config.llm.model.rsplit('/', 1)[-1]}_{create_safe_filename(question)}_{config.cores}"
    AGENT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    dual_console = DualConsole(AGENT_LOG_DIR / f"{output_dir_base}.log")
    agent = create_agent(llm=config.llm)
    agent.logger.console = dual_console
    agent.monitor.logger.console = dual_console

    output_dir = RAW_METRIC_DIR / output_dir_base
    output_dir.mkdir(parents=True, exist_ok=True)
    with DCGMI(output_dir=output_dir):
        with SAR(output_dir=output_dir):
            with VLLMMonitor(output_dir=output_dir):
                run_result = agent.run(question, return_full_result=True)

    print(f"Got this answer: {run_result.output}")

    header = ["node_id", "plan_start", "plan_end", "action_start", "action_end"]

    STEP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(STEP_LOG_DIR / f"{output_dir_base}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for step in run_result.steps:
            if "timing" not in step:
                continue

            row = {h: "" for h in header}
            start_time = step["timing"]["start_time"]
            end_time = step["timing"]["end_time"]

            if "plan" in step:
                row["node_id"] = "plan"
                row["plan_start"] = start_time
                row["plan_end"] = end_time
            else:
                row["node_id"] = f"action_{step['step_number']}"
                row["action_start"] = start_time
                row["action_end"] = end_time

            writer.writerow(row)


if __name__ == "__main__":
    main()
