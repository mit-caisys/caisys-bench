import os
import threading
import time

from .proc import Proc


class VLLM_METRICS:
    def __init__(self, output_dir=None):
        def _get_output_filepath():
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = f"vllm-metrics-{timestamp}.log"
            if os.path.exists(filepath):
                os.remove(filepath)
            return filepath

        # Minimum is 1s
        self.refresh_freq_s = 1
        self.vllm_metrics_output_filepath = _get_output_filepath()
        if output_dir:
            self.vllm_metrics_output_filepath = (
                f"{output_dir}/{self.vllm_metrics_output_filepath}"
            )

        self.vllm_metrics_command = [
            "bash",
            "-c",
            f"while true; do curl -s http://0.0.0.0:8000/metrics; sleep {self.refresh_freq_s}; done",
        ]

        self.vllm_metrics_output_file = open(self.vllm_metrics_output_filepath, "w")
        self.vllm_metrics_thread = None
        self.keep_alive_freq_s = 1
        self.is_running = False
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        print(f"Starting VLLM METRICS [freq = {self.refresh_freq_s} s] ...")
        self.is_running = True
        self.vllm_metrics_thread = threading.Thread(
            target=VLLM_METRICS.run,
            args=(
                self.vllm_metrics_command,
                self.vllm_metrics_output_file,
                self.keep_alive_freq_s,
                lambda: self.is_running,
            ),
        )
        self.vllm_metrics_thread.start()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.start_time = timestamp
        print(f"VLLM METRICS started at: {timestamp}")

    def __exit__(self, exc_type, exc_value, exc_tb):
        print("Stopping VLLM METRICS...")
        self.is_running = False
        self.vllm_metrics_thread.join()
        self.vllm_metrics_output_file.close()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.end_time = timestamp
        with open(self.vllm_metrics_output_filepath, "a") as file:
            file.write(f"property=start_time={self.start_time}\n")
            file.write(f"property=end_time={self.end_time}\n")
            file.write(f"property=refresh_freq_s={self.refresh_freq_s}\n")

        print(f"VLLM METRICS output saved in: {self.vllm_metrics_output_filepath}")
        print(f"VLLM METRICS stopped at: {timestamp}")

    @staticmethod
    def run(command, output_file, keep_alive_freq_s, is_running_fn):
        proc = Proc(command, output_file)
        proc.start()
        while is_running_fn():
            time.sleep(keep_alive_freq_s)
            if not proc.check():
                proc.restart()
        proc.stop()