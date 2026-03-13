import os
import threading
import time

from .proc import Proc


class SAR:
    def __init__(self, output_dir=None):
        def _get_output_filepath(op=""):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            if op != "":
                op = f"{op}-"
            filepath = f"sar-{op}{timestamp}.log"
            if os.path.exists(filepath):
                os.remove(filepath)
            return filepath

        # Minimum is 1s
        self.refresh_freq_s = 1
        self.cpu_mon_output_filepath = _get_output_filepath(op="cpu")
        self.mem_mon_output_filepath = _get_output_filepath(op="mem")
        self.nw_mon_output_filepath = _get_output_filepath(op="nw")
        if output_dir:
            self.cpu_mon_output_filepath = (
                f"{output_dir}/{self.cpu_mon_output_filepath}"
            )
            self.mem_mon_output_filepath = (
                f"{output_dir}/{self.mem_mon_output_filepath}"
            )
            self.nw_mon_output_filepath = f"{output_dir}/{self.nw_mon_output_filepath}"
        self.cpu_mon_command = f"sar -P ALL -u {self.refresh_freq_s}".split(" ")
        self.mem_mon_command = f"sar -P ALL -r {self.refresh_freq_s}".split(" ")
        self.nw_mon_command = f"sar -P ALL -n DEV {self.refresh_freq_s}".split(" ")
        self.cpu_mon_output_file = open(self.cpu_mon_output_filepath, "w")
        self.mem_mon_output_file = open(self.mem_mon_output_filepath, "w")
        self.nw_mon_output_file = open(self.nw_mon_output_filepath, "w")
        self.cpu_mon_thread = None
        self.mem_mon_thread = None
        self.nw_mon_thread = None
        self.keep_alive_freq_s = 1
        self.is_running = False

    def __enter__(self):
        print(f"Starting SAR [freq = {self.refresh_freq_s} s] ...")
        self.is_running = True
        self.cpu_mon_thread = threading.Thread(
            target=SAR.run,
            args=(
                self.cpu_mon_command,
                self.cpu_mon_output_file,
                self.keep_alive_freq_s,
                lambda: self.is_running,
            ),
        )
        self.mem_mon_thread = threading.Thread(
            target=SAR.run,
            args=(
                self.mem_mon_command,
                self.mem_mon_output_file,
                self.keep_alive_freq_s,
                lambda: self.is_running,
            ),
        )
        self.nw_mon_thread = threading.Thread(
            target=SAR.run,
            args=(
                self.nw_mon_command,
                self.nw_mon_output_file,
                self.keep_alive_freq_s,
                lambda: self.is_running,
            ),
        )
        self.cpu_mon_thread.start()
        self.mem_mon_thread.start()
        self.nw_mon_thread.start()
        time.strftime("%Y%m%d-%H%M%S")

    def __exit__(self, exc_type, exc_value, exc_tb):
        print("Stopping SAR...")
        self.is_running = False
        self.cpu_mon_thread.join()
        self.mem_mon_thread.join()
        self.nw_mon_thread.join()
        self.cpu_mon_output_file.close()
        self.mem_mon_output_file.close()
        self.nw_mon_output_file.close()
        print(f"SAR CPU trace output saved in: {self.cpu_mon_output_filepath}")
        print(f"SAR Memory trace output saved in: {self.mem_mon_output_filepath}")
        print(f"SAR Network trace output saved in: {self.nw_mon_output_filepath}")

    @staticmethod
    def run(command, output_file, keep_alive_freq_s, is_running_fn):
        proc = Proc(command, output_file)
        proc.start()
        while is_running_fn():
            time.sleep(keep_alive_freq_s)
            if not proc.check():
                proc.restart()
        proc.stop()
