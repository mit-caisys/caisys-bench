import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from monitoring import DCGMI
from monitoring import SAR

sys.path.append(str(Path(__file__).resolve().parents[1]))
from monitoring import VLLM_METRICS, VLLMMonitor 

if len(sys.argv) > 1:
    output_folder = sys.argv[1]
else:
    print("Warning: No output directory specified. Using current directory.")
    output_folder = "."


FRAME_COUNTS = [10]
TRANSCRIPTION_OPTIONS = [True]
BASE_COMMAND = ["python", "langgraph_multimodal_workflow.py", "--config"]


def run_experiments():
    """
    Generates configuration files and runs experiments based on a base config.
    """
    try:
        script_dir = Path(__file__).parent
        with open(script_dir / "config.json", "r") as f:
            base_config = json.load(f)
    except FileNotFoundError:
        print(
            "Error: base_config.json not found. Make sure it's in the same directory."
        )
        return
    configs_dir = os.path.join(output_folder, "generated_configs")
    os.makedirs(configs_dir, exist_ok=True)

    all_results = []

    print("Starting experiment runs...")

    for frames in FRAME_COUNTS:
        for transcription in TRANSCRIPTION_OPTIONS:
            # Create a deep copy to avoid modifying the original dictionary
            new_config = copy.deepcopy(base_config)

            # Modify the configuration for the current experiment
            new_config["workflow_settings"]["default_num_frames"] = frames
            new_config["inputs"]["video_mme_input"]["transcription"] = transcription

            # Create a descriptive filename for the new config
            trans_str = "with_transcription" if transcription else "no_transcription"
            config_filename = os.path.join(
                configs_dir, f"config_frames_{frames}_{trans_str}.json"
            )

            # Save the new configuration to a file
            with open(config_filename, "w") as f:
                json.dump(new_config, f, indent=2)

            print("-" * 50)
            print(
                f"Running experiment: Frames = {frames}, Transcription = {transcription}"
            )
            print(f"Generated config file: {config_filename}")

            # Construct the full command to run
            command = BASE_COMMAND + [config_filename]

            output_dir_logs = os.path.join(output_folder, "monitoring_logs")
            os.makedirs(output_dir_logs, exist_ok=True)

            try:
                with DCGMI(output_dir=output_dir_logs):
                    with SAR(output_dir=output_dir_logs):
                        with VLLMMonitor(output_dir=output_dir_logs):
                            result = subprocess.run(
                                command, check=True, capture_output=True, text=True
                            )
                # print(result.stdout)
                print("Experiment completed successfully.")
                # You can optionally print stdout or stderr from your script
                # print("Output:\n", result.stdout)
                last_line = result.stdout.strip().split("\n")[-1]
                run_result = json.loads(last_line)

                token_stats = run_result.get("complete_token_stats", [])
                latency_stats = run_result.get("individual_latencies", [])
                end_to_end_latencies = run_result.get(
                    "complete_end_to_end_latencies", []
                )
                end_to_end_latencies_load_gen = run_result.get(
                    "complete_end_to_end_latencies_load_gen", []
                )
                correctness = run_result.get("complete_correctness")

                if not latency_stats:
                    print(
                        "Warning: No individual latency stats found for this run. Skipping CSV generation."
                    )
                    continue

                num_requests = len(latency_stats)
                print(f"Found data for {num_requests} individual requests.")

                per_request_results = []
                for i in range(num_requests):
                    current_latencies = latency_stats[i]
                    current_tokens = token_stats[i]
                    current_correctness = correctness[i]

                    row_data = {
                        "request_id": i + 1,
                        "encode_videos_start": current_latencies.get(
                            "encode-videos"
                        ).get("start_time"),
                        "encode_videos_end": current_latencies.get("encode-videos").get(
                            "end_time"
                        ),
                        "speech_to_text_start": (
                            current_latencies.get("speech_to_text").get("start_time")
                            if current_latencies.get("speech_to_text")
                            else None
                        ),
                        "speech_to_text_end": (
                            current_latencies.get("speech_to_text").get("end_time")
                            if current_latencies.get("speech_to_text")
                            else None
                        ),
                        "multimodal_model_start": current_latencies.get(
                            "generate_answer_vllm"
                        ).get("start_time"),
                        "multimodal_model_end": current_latencies.get(
                            "generate_answer_vllm"
                        ).get("end_time"),
                        "end_to_end_start": end_to_end_latencies[i].get("start_time"),
                        "end_to_end_end": end_to_end_latencies[i].get("end_time"),
                        "end_to_end_load_gen_start": end_to_end_latencies_load_gen[
                            i
                        ].get("start_time"),
                        "end_to_end_load_gen_end": end_to_end_latencies_load_gen[i].get(
                            "end_time"
                        ),
                        "prompt_tokens": current_tokens.get("prompt_tokens"),
                        "completion_tokens": current_tokens.get("completion_tokens"),
                        "total_tokens": current_tokens.get("total_tokens"),
                        "correct": current_correctness,
                        # # Summary stats for the entire run (repeated for each row for context)
                        # "run_accuracy": run_result.get("accuracy"),
                        # "run_total_correct": run_result.get("total_correct"),
                        # "run_total_questions": run_result.get("total_questions"),
                        # "run_total_batch_time": run_result.get("total_batch_processing_time"),
                    }
                    per_request_results.append(row_data)

                results_filename = f"frames-{frames}_transcription-{transcription}.csv"
                results_filepath = os.path.join(output_folder, results_filename)

                print(f"Writing per-request results to {results_filepath}...")
                if per_request_results:
                    try:
                        with open(results_filepath, "w", newline="") as csvfile:
                            fieldnames = list(per_request_results[0].keys())
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(per_request_results)
                        print(f"Results successfully saved to {results_filepath}.")
                    except IOError as e:
                        print(f"Failed to write results to CSV. Error: {e}")
                else:
                    print("No result rows were generated to write.")

            except FileNotFoundError:
                print(
                    f"Error: Command not found. Is '{' '.join(BASE_COMMAND)}' correct?"
                )
                break
            except subprocess.CalledProcessError as e:
                print(f"Experiment failed with exit code {e.returncode}.")
                print("Error output:\n", e.stderr)
            except json.JSONDecodeError:
                print(
                    "Error: Failed to decode JSON from the last line of script output."
                )
                print("Last line was:", last_line)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    #             token_stats = run_result.get("complete_token_stats")
    #             avg_tokens = sum([token_stat.get("total_tokens") for token_stat in token_stats])/len(token_stats)
    #             avg_prompt_tokens = sum([token_stat.get("prompt_tokens") for token_stat in token_stats])/len(token_stats)
    #             avg_completion_tokens = sum([token_stat.get("completion_tokens") for token_stat in token_stats])/len(token_stats)
    #             latency_stats = run_result.get("individual_latencies")

    #             encode_videos_node_latencies = [latency_stat.get("encode-videos") for latency_stat in latency_stats]
    #             speech_to_text_node_latencies = [latency_stat.get("speech_to_text") for latency_stat in latency_stats] if latency_stats[0].get("speech_to_text") else None
    #             model_node_latencies = [latency_stat.get("generate_answer_vllm") for latency_stat in latency_stats]

    #             encode_videos_node_str = ";".join(map(str, encode_videos_node_latencies))
    #             speech_to_text_node_str = ";".join(map(str, speech_to_text_node_latencies)) if speech_to_text_node_latencies else ""
    #             model_node_str = ";".join(map(str, model_node_latencies))

    #             encode_videos_node_avg_latency = sum(encode_videos_node_latencies)/len(latency_stats)
    #             speech_to_text_node_avg_latency = sum(speech_to_text_node_latencies)/len(latency_stats) if speech_to_text_node_latencies else 0
    #             model_node_avg_latency = sum(model_node_latencies)/len(latency_stats)

    #             end_to_end_latencies = run_result.get("complete_end_to_end_latencies")
    #             end_to_end_latencies_str = ";".join(map(str, end_to_end_latencies))

    #             end_to_end_latencies_load_gen = run_result.get("complete_end_to_end_latencies")
    #             end_to_end_latencies_load_gen_str = ";".join(map(str, end_to_end_latencies)) if end_to_end_latencies_load_gen else ""

    #             # Store the parameters and results for this run
    #             all_results.append({
    #                 "frames": frames,
    #                 "transcription": transcription,
    #                 "accuracy": run_result.get("accuracy"),
    #                 "total_correct": run_result.get("total_correct"),
    #                 "total_questions": run_result.get("total_questions"),
    #                 "total_batch_processing_time": run_result.get("total_batch_processing_time"),
    #                 "avg_latency_per_request": run_result.get("total_batch_processing_time")/len(latency_stats),
    #                 "encode_videos_node_avg_latency": encode_videos_node_avg_latency,
    #                 "speech_to_text_node_avg_latency" : speech_to_text_node_avg_latency,
    #                 "model_node_avg_latency": model_node_avg_latency,
    #                 "avg_prompt_tokens_per_request": avg_prompt_tokens,
    #                 "avg_completion_tokens_per_request": avg_completion_tokens,
    #                 "avg_tokens_per_request": avg_tokens,
    #                 "encode_videos_node_latencies": encode_videos_node_str,
    #                 "speech_to_text_node_latencies": speech_to_text_node_str,
    #                 "model_node_latencies": model_node_str,
    #                 "end_to_end_latencies": end_to_end_latencies_str,
    #                 "end_to_end_latencies_load_gen": end_to_end_latencies_load_gen_str
    #             })
    #         except FileNotFoundError:
    #             print(f"Error: Command not found. Is '{' '.join(BASE_COMMAND)}' correct?")
    #             return
    #         except subprocess.CalledProcessError as e:
    #             print(f"Experiment failed with exit code {e.returncode}.")
    #             print("Error output:\n", e.stderr)
    #         except Exception as e:
    #             print(f"An unexpected error occurred: {e}")

    # results_filename = os.path.join(output_folder, "experiment_results.csv")
    # print("-" * 50)
    # print(f"Writing all results to {results_filename}...")

    # try:
    #     with open(results_filename, 'w', newline='') as csvfile:
    #         fieldnames = ["frames", "transcription", "accuracy", "total_correct", "total_questions",
    #                        "total_batch_processing_time", "avg_latency_per_request", "avg_prompt_tokens_per_request",
    #                        "encode_videos_node_avg_latency", "speech_to_text_node_avg_latency", "model_node_avg_latency",
    #                          "avg_completion_tokens_per_request", "avg_tokens_per_request",
    #                          "encode_videos_node_latencies", "speech_to_text_node_latencies", "model_node_latencies",
    #                          "end_to_end_latencies", "end_to_end_latencies_load_gen"]
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #         writer.writeheader()
    #         writer.writerows(all_results)

    #     print(f"Results successfully saved to {results_filename}.")
    # except Exception as e:
    #     print(f"Failed to write results to CSV. Error: {e}")

    print("-" * 50)
    print("All experiments finished.")


if __name__ == "__main__":
    run_experiments()

