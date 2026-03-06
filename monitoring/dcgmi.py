"""
DCGMI (Data Center GPU Manager) monitoring module.

Context manager for running NVIDIA DCGM diagnostic mode commands
to collect GPU metrics.
"""

import os
import threading
import time

from .proc import Proc

LOCAL_NODE = "local"


class DCGMI:
    def __init__(self, output_dir=None, remote_nodes=None):
        def _get_output_filepath(node):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = f"dcgmi-{node}-{timestamp}.log"
            if os.path.exists(filepath):
                os.remove(filepath)
            return filepath

        # Comma-separated list (command: dcgmi dmon -l)
        fields = [
            155,
            157,
            203,
            204,
            525,
            1002,
            1003,
            1004,
            1005,
            1006,
            1007,
            1008,
            1009,
            1010,
            1011,
            1012,
            1100,
        ]
        self.fields = ",".join([str(field) for field in fields])
        # Minimum is 100ms
        self.refresh_freq_ms = 100
        dcgmi_command = [
            "dcgmi",
            "dmon",
            "-e",
            f"{self.fields}",
            "-d",
            f"{self.refresh_freq_ms}",
        ]
        local_node_output_filepath = _get_output_filepath(LOCAL_NODE)
        if output_dir:
            local_node_output_filepath = f"{output_dir}/{local_node_output_filepath}"
        self.commands = [dcgmi_command]
        self.nodes = [LOCAL_NODE]
        self.output_filepaths = [local_node_output_filepath]
        self.output_files = [open(local_node_output_filepath, "w")]
        if remote_nodes is not None:
            for remote_node in remote_nodes:
                command = dcgmi_command + ["--host", remote_node]
                output_filepath = _get_output_filepath(remote_node)
                if output_dir:
                    output_filepath = f"{output_dir}/{output_filepath}"
                self.output_filepaths.append(output_filepath)
                self.nodes.append(remote_node)
                self.commands.append(command)
                self.output_files.append(open(output_filepath, "w"))
        self.num_nodes = len(self.nodes)
        self.threads = []
        self.keep_alive_freq_s = 1
        self.is_running = False
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        print(f"Starting DCGMI [freq = {self.refresh_freq_ms} ms] ...")
        self.is_running = True
        for i in range(self.num_nodes):
            self.threads.append(
                threading.Thread(
                    target=DCGMI.run,
                    args=(
                        self.commands[i],
                        self.output_files[i],
                        self.keep_alive_freq_s,
                        lambda: self.is_running,
                    ),
                )
            )
        for thread in self.threads:
            thread.start()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.start_time = timestamp
        print(f"DCGMI started at: {timestamp}")

    def __exit__(self, exc_type, exc_value, exc_tb):
        print("Stopping DCGMI...")
        self.is_running = False
        for thread in self.threads:
            thread.join()
        for output_file in self.output_files:
            output_file.close()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.end_time = timestamp
        for i in range(self.num_nodes):
            output_filepath = self.output_filepaths[i]
            node = self.nodes[i]
            with open(output_filepath, "a") as file:
                file.write(f"property=start_time={self.start_time}\n")
                file.write(f"property=end_time={self.end_time}\n")
                file.write(f"property=refresh_freq_ms={self.refresh_freq_ms}\n")
                file.write(f"property=node={node}")
            print(f"DCGMI output saved in: {output_filepath}")
        print(f"DCGMI stopped at: {timestamp}")

    @staticmethod
    def run(command, output_file, keep_alive_freq_s, is_running_fn):
        print("Running command:", " ".join(command))
        proc = Proc(command, output_file)
        proc.start()
        while is_running_fn():
            time.sleep(keep_alive_freq_s)
            if not proc.check():
                proc.restart()
        proc.stop()
