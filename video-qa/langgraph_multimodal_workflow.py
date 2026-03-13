import argparse
import asyncio
import json
import logging
import os
import time
import random

from datasets import load_dataset
from langchain.globals import set_debug
from langgraph.graph import END, StateGraph
from load_gen import Request, poisson_load_generator
from video_processing_nodes import (
    GraphState,
    encode_videos,
    generate_answer,
    generate_answer_vllm,
    route_images,
    speech_to_text_transcription,
    transcription_route,
    warmup_models,
)

set_debug(False)

log_dir = "data/logs"

log_file = os.path.join(log_dir, "vide-app.log")

logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def track_latency(node_name: str):
    def decorator(func):
        def wrapper(state: GraphState) -> GraphState:
            start_time = time.time()
            result_state = func(state)
            end_time = time.time()

            if "latency_tracker" not in result_state:
                result_state["latency_tracker"] = {}

            result_state["latency_tracker"][node_name] = {
                "start_time": start_time,
                "end_time": end_time,
            }

            return result_state

        return wrapper

    return decorator


def run_workflows(config: dict):
    workflow = StateGraph(GraphState)

    # Add the nodes that perform work
    # workflow.add_node("encoder", encode_images)
    workflow.add_node("video_encoder", encode_videos)
    workflow.add_node("speech_to_text", speech_to_text_transcription)

    if config["model_config"]["active_model"] == "vllm":
        workflow.add_node("model", generate_answer_vllm)
    elif config["model_config"]["active_model"] == "openai":
        workflow.add_node("model", generate_answer)
    else:
        raise ValueError("active model not configured to a supported model")
    workflow.set_conditional_entry_point(
        route_images,
        {
            # "encoder": "encoder",
            "video_encoder": "video_encoder",
            # "model": "model"
        },
    )

    # Define the remaining edges in the graph
    # workflow.add_edge("encoder", "model") # After encoding, the next step is the model
    workflow.add_conditional_edges(
        "video_encoder",
        transcription_route,
        {"speech_to_text": "speech_to_text", "model": "model"},
    )
    workflow.add_edge("speech_to_text", "model")
    workflow.add_edge("model", END)

    app = workflow.compile()

    try:
        with open("warmup_config.json", "r") as f:
            config_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {args.config}")
        exit(1)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the configuration file: {args.config}")
        exit(1)

    # doing warmup for models before running anything to avoid cold start
    warmup_models(app, config_data)

    if config["workflow_settings"]["active_workflows"]["images_from_urls"]:
        # --- RUN WORKFLOW 1: MULTIPLE IMAGES FROM URLS ---
        logging.info("---RUNNING WORKFLOW: IMAGES FROM URLS---")

        # We define the initial state as a dictionary and pass it to invoke.
        url_input = config["inputs"]["image_url_input"]
        # Each invoke call is independent and starts with the state we provide.
        final_state_url = app.invoke(url_input)
        print("\n---FINAL ANSWER (from URLs)---")
        print(final_state_url.get("answer"))
        print("=" * 50)

    if config["workflow_settings"]["active_workflows"]["images_from_local"]:

        # --- RUN WORKFLOW 2: MULTIPLE LOCAL IMAGES ---
        print("\n" + "=" * 50)
        logging.info("---RUNNING WORKFLOW: LOCAL IMAGES---")

        local_input = config["inputs"]["local_image_input"]
        final_state_local = app.invoke(local_input)
        print("\n---FINAL ANSWER (from Local Paths)---")
        print(final_state_local.get("answer"))
        print("=" * 50)

    if config["workflow_settings"]["active_workflows"]["video_from_local"]:

        print("\n" + "=" * 50)
        logging.info("---RUNNING WORKFLOW 3: Video Workflow---")

        local_video_input = config["inputs"]["local_video_input"]

        final_state_video = app.invoke(local_video_input)

        print("\n---FINAL ANSWER (from Local Paths)---")
        print(final_state_video.get("answer"))
        print("=" * 50)

        print("\n" + "=" * 50)

    if config["workflow_settings"]["active_workflows"]["video_mme_dataset"]:
        logging.info("---RUNNING WORKFLOW 4: Video-MME Workflow VLLM/OPENAI---")

        real_answers = []

        video_paths = [
            f for f in os.listdir(config["inputs"]["video_mme_input"]["videos_path"])
        ]
        video_ids = [os.path.splitext(path)[0] for path in video_paths][:1]

        video_inputs = []

        ds = load_dataset("lmms-lab/Video-MME")
        short_ds = ds.filter(lambda example: example["duration"] == "medium")

        NUM_DUMMY_REQUESTS = 3  # How many cache hits to test per video
        DUMMY_CHOICE = ["A"]


        for row in short_ds["test"]:
            if row["videoID"] in video_ids:
                idx = video_ids.index(row["videoID"])
                video_input = {
                    "video_path": config["inputs"]["video_mme_input"]["videos_path"]
                    + "/"
                    + video_paths[idx],
                    "question": f'Based on the materials provided answer the question by picking an option. \n Question: {row["question"]} \n Options: {row["options"]}',
                    "config": config,
                }
                real_answers.append(row["answer"])
                video_inputs.append(video_input)

        combined = list(zip(video_inputs, real_answers))

        # random.seed(42) 
        # random.shuffle(combined)

        video_inputs, real_answers = zip(*combined)
        
        video_inputs = list(video_inputs)
        real_answers = list(real_answers)

        complete_end_to_end_latencies_load_gen = []

        if config["inputs"]["video_mme_input"]["poisson_load_gen"]:
            poisson_rate = config["inputs"]["video_mme_input"]["poisson_lambda"]

            total_batch_start_time = time.time()

            load_gen_results = asyncio.run(
                poisson_load_generator(app, video_inputs, poisson_rate)
            )

            if not isinstance(load_gen_results, list):
                raise ValueError("Unexpected type")

            if all(isinstance(item, Request) for item in load_gen_results):
                final_state_data = [request.result for request in load_gen_results]
                # complete_end_to_end_latencies_load_gen = [request.end_time - request.start_time for request in load_gen_results]
                complete_end_to_end_latencies_load_gen = [
                    {"start_time": request.start_time, "end_time": request.end_time}
                    for request in load_gen_results
                ]

            total_batch_end_time = time.time()
        else:
            final_state_data = []
            total_batch_start_time = time.time()
            for i, input_item in enumerate(video_inputs):
                print(input_item)
                req_start = time.time()
                batch_result = app.batch([input_item])
                final_state_data.extend(batch_result)
                req_end = time.time()
                complete_end_to_end_latencies_load_gen.append({
                    "start_time": req_start,
                    "end_time": req_end
                })
            total_batch_end_time = time.time()

        total_batch_latency = total_batch_end_time - total_batch_start_time

        all_latencies = []

        count_correct = 0
        complete_token_stats = []
        complete_end_to_end_latencies = []
        complete_correctness = []

        for i in range(0, len(final_state_data)):

            if final_state_data[i].get("answer").answer.value == real_answers[i]:
                count_correct += 1
                complete_correctness.append(True)
            else:
                complete_correctness.append(False)

            token_stats = final_state_data[i]["token_stats"]
            end_to_end_times = {
                "start_time": final_state_data[i]["start_time"],
                "end_time": final_state_data[i]["end_time"],
            }

            complete_token_stats.append(token_stats)

            complete_end_to_end_latencies.append(end_to_end_times)
            all_latencies.append(final_state_data[i].get("latency_tracker", {}))
        total_questions = len(final_state_data)
        accuracy = count_correct / len(final_state_data)
        print("\n---FINAL ACCURACY (from Dataset)---")
        print(accuracy)
        print("\n---TOTAL NUMBER OF QUESTIONS (from Dataset)---")
        print(total_questions)

        print("=" * 50)

        results_dict = {
            "accuracy": accuracy,
            "total_correct": count_correct,
            "complete_correctness": complete_correctness,
            "total_batch_processing_time": total_batch_latency,
            "individual_latencies": all_latencies,
            "complete_token_stats": complete_token_stats,
            "complete_end_to_end_latencies": complete_end_to_end_latencies,
            "complete_end_to_end_latencies_load_gen": complete_end_to_end_latencies_load_gen,
        }

        print(json.dumps(results_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run video/image QA wosrkflows.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file for the experiment.",
    )
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {args.config}")
        exit(1)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the configuration file: {args.config}")
        exit(1)

    # # Set OpenAI API key from config if available
    # api_key = config_data.get("api_keys", {}).get("openai_api_key")
    # if api_key and api_key != "YOUR_OPENAI_API_KEY_HERE":
    #     os.environ["OPENAI_API_KEY"] = api_key

    # if not os.getenv("OPENAI_API_KEY"):
    #     print("Warning: OPENAI_API_KEY environment variable is not set.")

    # Run the main logic with the loaded configuration
    run_workflows(config_data)

